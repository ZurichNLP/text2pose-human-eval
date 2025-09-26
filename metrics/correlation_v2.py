import argparse
import pandas as pd
import os
import re
import json
import numpy as np
import glob  # for file matching

from scipy.stats import pearsonr, spearmanr
import warnings

# Ensure all columns are printed without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
    # Determine the directory where the script is located (for finding metrics files)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------
    # STEP 1: Read and filter human scores
    # ------------------------------
    parser = argparse.ArgumentParser(description="Correlation Study: Aggregate Human Scores and Join Metrics")
    parser.add_argument(
        '--human_score',
        type=str,
        default='./human_evaluation/batches_text2pose/results/aggregated.csv',
        help='Path to the human score CSV file (default: ./human_evaluation/batches_text2pose/results/aggregated.csv).'
    )
    parser.add_argument(
        '--exclude_evaluators',
        nargs='+',
        default=['Lisa0'],
        help='List of evaluators to exclude (default: [\'Lisa0\']).'
    )
    parser.add_argument(
        '--systems',
        nargs='+',
        default=['sockeye', 'sign_mt', 'sign_mt_v2', 'ref'],
        help='List of systems to include (default: [\'sockeye\', \'sign_mt\', \'sign_mt_v2\']).'
    )
    args = parser.parse_args()

    # Read the human score CSV
    df = pd.read_csv(args.human_score)
    print("=== Original DataFrame ===")
    print(df)
    
    # Filter out rows with evaluators in the exclude list
    df = df[~df['username'].isin(args.exclude_evaluators)]
    # Filter to include only rows with systems in the provided list
    df = df[df['system'].isin(args.systems)]
    print("\n=== Filtered DataFrame ===")
    print(df)
    
    # Add the new 'direction' column as a combination of srclang and trglang
    df['direction'] = df['srclang'] + '->' + df['trglang']

    # ----------------------------------------
    # NEW STEP: Inter-evaluator correlations summary table (non-ref only)
    # ----------------------------------------

    # restrict to non-ref
    non_ref = df[df['system'] != 'ref']

    # count annotators per segment
    key = ['system','itemid','direction']
    counts = (
        non_ref
        .groupby(key)['username']
        .nunique()
        .reset_index(name='n_annot')
    )

    def compute_per_eval(df_subset):
        """
        For the given subset (can be non_ref or ref_only),
        recompute counts, then return evaluator -> {pearson, spearman, n}
        """
        key = ['system','itemid','direction']
        # recompute counts for this subset
        counts_sub = (
            df_subset
            .groupby(key)['username']
            .nunique()
            .reset_index(name='n_annot')
        )

        results = {}
        for ev in df_subset['username'].unique():
            ev_df = (
                df_subset[df_subset['username'] == ev]
                .merge(counts_sub, on=key, how='left')
            )
            # drop & warn about any segments they alone annotated
            sole = ev_df[ev_df['n_annot'] == 1]
            if not sole.empty:
                warnings.warn(
                    f"Evaluator '{ev}' has {len(sole)} segment(s) with no co-annotators; skipping those."
                )
            ev_df = ev_df[ev_df['n_annot'] >= 2]
            if ev_df.empty:
                continue

            # build the “others” mean
            others = df_subset[df_subset['username'] != ev]
            others_mean = (
                others
                .groupby(key)['score']
                .mean()
                .rename('score_others')
                .reset_index()
            )

            paired = (
                ev_df[key + ['score']]
                .rename(columns={'score':'score_self'})
                .merge(others_mean, on=key, how='inner')
            )

            n_seg = len(paired)
            r, _   = pearsonr(paired['score_self'], paired['score_others'])
            rho, _ = spearmanr(paired['score_self'], paired['score_others'])
            results[ev] = {'pearson': r, 'spearman': rho, 'n': n_seg}

        return results

    # 1) overall
    overall = compute_per_eval(non_ref)

    # 2) by system
    by_system = {
        sys: compute_per_eval(non_ref[non_ref['system'] == sys])
        for sys in non_ref['system'].unique()
    }

    # 3) by direction
    by_direction = {
        d: compute_per_eval(non_ref[non_ref['direction'] == d])
        for d in non_ref['direction'].unique()
    }

    # build the summary table
    rows = []

    # — per‐evaluator
    for ev, scores in overall.items():
        rows.append({
            'Category':    ev,
            'Pearson':     scores['pearson'],
            'Spearman':    scores['spearman'],
            'NumSegments': scores['n'],
        })

    # — overall mean
    if overall:
        mean_p = np.mean([v['pearson']  for v in overall.values()])
        mean_s = np.mean([v['spearman'] for v in overall.values()])
        mean_n = np.mean([v['n']        for v in overall.values()])
        rows.append({
            'Category':    'Overall mean',
            'Pearson':     mean_p,
            'Spearman':    mean_s,
            'NumSegments': mean_n,
        })

    # — system‐level (mean over evaluators)
    for sys, ev_dict in by_system.items():
        if not ev_dict:
            continue
        p = np.mean([v['pearson']  for v in ev_dict.values()])
        s = np.mean([v['spearman'] for v in ev_dict.values()])
        n = np.mean([v['n']        for v in ev_dict.values()])
        rows.append({
            'Category':    f'System: {sys}',
            'Pearson':     p,
            'Spearman':    s,
            'NumSegments': n,
        })

    # — direction‐level
    for d, ev_dict in by_direction.items():
        if not ev_dict:
            continue
        p = np.mean([v['pearson']  for v in ev_dict.values()])
        s = np.mean([v['spearman'] for v in ev_dict.values()])
        n = np.mean([v['n']        for v in ev_dict.values()])
        rows.append({
            'Category':    f'Direction: {d}',
            'Pearson':     p,
            'Spearman':    s,
            'NumSegments': n,
        })

    summary_df = pd.DataFrame(rows).set_index('Category')
    sd_rows = [
        'Overall mean',
        'System: sign_mt',
        'System: sockeye',
        'System: sign_mt_v2',
        'Direction: deu->sgg',
        'Direction: ita->ise',
        'Direction: fra->fsl',
    ]
    # compute SD only over those rows
    summary_df.loc['SD'] = summary_df.loc[sd_rows].std()
    print("\n=== Inter-evaluator Correlation Summary (non-ref) ===")
    print(summary_df)

    # ----------------------------------------
    # NEW STEP: Inter-evaluator correlations for ref only
    # ----------------------------------------
    ref_only = df[df['system'] == 'ref']

    # 1) overall for ref
    ref_overall = compute_per_eval(ref_only)

    # 2) by direction for ref
    ref_by_direction = {
        d: compute_per_eval(ref_only[ref_only['direction'] == d])
        for d in ref_only['direction'].unique()
    }

    # Build the ref summary rows
    ref_rows = []

    # — per‐evaluator
    for ev, scores in ref_overall.items():
        ref_rows.append({
            'Category':    ev,
            'Pearson':     scores['pearson'],
            'Spearman':    scores['spearman'],
            'NumSegments': scores['n'],
        })

    # — overall mean for ref
    if ref_overall:
        mean_p = np.mean([v['pearson']  for v in ref_overall.values()])
        mean_s = np.mean([v['spearman'] for v in ref_overall.values()])
        mean_n = np.mean([v['n']        for v in ref_overall.values()])
        ref_rows.append({
            'Category':    'Overall mean (ref)',
            'Pearson':     mean_p,
            'Spearman':    mean_s,
            'NumSegments': mean_n,
        })

    # — direction‐level for ref
    for d, ev_dict in ref_by_direction.items():
        if not ev_dict:
            continue
        p = np.mean([v['pearson']  for v in ev_dict.values()])
        s = np.mean([v['spearman'] for v in ev_dict.values()])
        n = np.mean([v['n']        for v in ev_dict.values()])
        ref_rows.append({
            'Category':    f'Direction: {d} (ref)',
            'Pearson':     p,
            'Spearman':    s,
            'NumSegments': n,
        })

    if ref_rows:
        ref_summary_df = pd.DataFrame(ref_rows).set_index('Category')
        ref_sd_rows = [
            'Overall mean (ref)',
            'Direction: deu->sgg (ref)',
            'Direction: ita->ise (ref)',
            'Direction: fra->fsl (ref)',
        ]
        ref_summary_df.loc['SD'] = ref_summary_df.loc[ref_sd_rows].std()
        print("\n=== Inter-evaluator Correlation Summary (ref) ===")
        print(ref_summary_df)
    else:
        print("No inter-evaluator correlations computed for ref (still no overlapping segments).")
    
    # Aggregate rows by taking the mean score for each (system, itemid) combination
    aggregated_df = df.groupby(['system', 'itemid', 'direction'], as_index=False)['score'].mean()
    print("\n=== Aggregated DataFrame (before joining metrics) ===")
    print(aggregated_df)
    
    # Print a summary of itemids for each system (count and range)
    print("\n=== ItemID Summary by System ===")
    for sys, group in aggregated_df.groupby('system'):
        count = group['itemid'].nunique()
        min_id = group['itemid'].min()
        max_id = group['itemid'].max()
        print(f"System: {sys} - Count: {count}, ItemID Range: {min_id} - {max_id}")
    
    # ------------------------------
    # STEP 2: Join CSV metrics (first source)
    # ------------------------------
    merged_dfs = []
    for sys in aggregated_df['system'].unique():
        df_sys = aggregated_df[aggregated_df['system'] == sys]
        metrics_file = os.path.join(script_dir, f"metrics.{sys}.csv")
        if os.path.exists(metrics_file):
            try:
                df_metrics = pd.read_csv(metrics_file)
                df_metrics_subset = df_metrics[['example_id', 'nMSE', 'nAPE', 'DTW', 'nDTW']]
                merged_sys = pd.merge(df_sys, df_metrics_subset, left_on='itemid', right_on='example_id', how='left')
                merged_sys.drop(columns='example_id', inplace=True)
                merged_dfs.append(merged_sys)
            except Exception as e:
                print(f"Error processing metrics file '{metrics_file}' for system '{sys}': {e}")
                merged_dfs.append(df_sys)
        else:
            print(f"Metrics file '{metrics_file}' not found for system '{sys}'.")
            merged_dfs.append(df_sys)
    
    aggregated_df_with_csv_metrics = pd.concat(merged_dfs, ignore_index=True)
    print("\n=== Aggregated DataFrame with CSV Metrics ===")
    print(aggregated_df_with_csv_metrics)

    # ------------------------------
    # STEP 3: Load and merge JSON metrics (second source)
    # ------------------------------
    json_file1 = os.path.join(script_dir, "metrics_back_translation.json")
    json_file2 = os.path.join(script_dir, "metrics_signmt_v2.json")
    
    with open(json_file1, 'r') as f:
        bt_data = json.load(f)
    with open(json_file2, 'r') as f:
        bt_data_2 = json.load(f)
    bt_data.update(bt_data_2)
    
    json_metrics_dfs = []
    for sys, metrics_list in bt_data.items():
        if len(metrics_list) != 1000:
            print(f"Warning: Expected 1000 entries for system '{sys}' in JSON, found {len(metrics_list)}")
        df_json = pd.DataFrame(metrics_list)
        df_json['itemid'] = df_json.index
        
        df_json['BLEU-4'] = df_json['bleu'].apply(
            lambda s: float(re.search(r"BLEU\s*=\s*([0-9.]+)", s).group(1)) if pd.notnull(s) and re.search(r"BLEU\s*=\s*([0-9.]+)", s) else None
        )
        df_json['chrF'] = df_json['chrf'].apply(
            lambda s: float(re.search(r"chrF2\s*=\s*([0-9.]+)", s).group(1)) if pd.notnull(s) and re.search(r"chrF2\s*=\s*([0-9.]+)", s) else None
        )
        df_json['BLEURT'] = df_json['bleurt'].astype(float)
        df_json['Likelihood'] = df_json['score'].astype(float)
        
        df_json = df_json[['itemid', 'BLEU-4', 'chrF', 'BLEURT', 'Likelihood']]
        df_json['system'] = sys
        json_metrics_dfs.append(df_json)
    
    if json_metrics_dfs:
        json_metrics_df = pd.concat(json_metrics_dfs, ignore_index=True)
    else:
        json_metrics_df = None
    
    if json_metrics_df is not None:
        final_df = pd.merge(aggregated_df_with_csv_metrics, json_metrics_df, on=['system', 'itemid'], how='left')
        print("\n=== Aggregated DataFrame with Metrics (CSV and JSON) ===")
        print(final_df)
    else:
        final_df = aggregated_df_with_csv_metrics.copy()
        print("No JSON metrics data found.")
    
    # ------------------------------
    # STEP 4: Merge SkeletonVAE metrics (third source)
    # ------------------------------
    # find all skeleton VAE norm CSVs
    skl_files = glob.glob(os.path.join(script_dir, "skeleton_vae_norm*.csv"))
    if skl_files:
        skl_merged = None
        for skl_file in skl_files:
            try:
                df_skl = pd.read_csv(skl_file).rename(columns={"sentence_id": "itemid"})
                if skl_merged is None:
                    skl_merged = df_skl
                else:
                    # inner join keeps only the 1000 common rows
                    skl_merged = pd.merge(skl_merged, df_skl, on="itemid", how="inner")
            except Exception as e:
                print(f"Error reading skeleton VAE file '{skl_file}': {e}")
        if skl_merged is not None:
            # pivot long: one row per (itemid, system)
            records = []
            for system in ['sign_mt', 'sign_mt_v2', 'sockeye']:
                val_col  = system
                path_col = f"{system}_path_norm"
                ref_col  = f"{system}_ref_norm"
                df_sys = skl_merged[['itemid', val_col, path_col, ref_col]].copy()
                df_sys['system'] = system
                df_sys = df_sys.rename(columns={
                    val_col:                'SkeletonVAE',
                    path_col:               'SkeletonVAE_path_norm',
                    ref_col:                'SkeletonVAE_ref_norm'
                })
                records.append(df_sys)
            skl_long_df = pd.concat(records, ignore_index=True)   # 3000 rows: 3 systems × 1000 items

            # merge into your final_df
            final_df = pd.merge(
                final_df,
                skl_long_df,
                on=['system', 'itemid'],
                how='left'
            )
            print("\n=== Final DataFrame with SkeletonVAE metrics ===")
            print(final_df)
        else:
            print("No valid skeleton VAE data found to merge.")
    else:
        print("No skeleton VAE files found matching pattern 'skeleton_vae_norm*.csv'.")
    
    # ------------------------------
    # STEP 5: Read & horizontally merge all Pose SKL files (fourth source)
    # ------------------------------
    pose_skl_files = glob.glob(os.path.join(script_dir, "pose_skl*.csv"))
    if pose_skl_files:
        pose_skl_merged = None
        for skl_file in pose_skl_files:
            try:
                df_skl = pd.read_csv(skl_file).rename(columns={"example_id": "itemid"})
                if pose_skl_merged is None:
                    pose_skl_merged = df_skl
                else:
                    pose_skl_merged = pd.merge(
                        pose_skl_merged,
                        df_skl,
                        on=['system', 'itemid'],
                        how='outer'
                    )
            except Exception as e:
                print(f"Error reading pose SKL file '{skl_file}': {e}")
        if pose_skl_merged is not None:
            # now join all SKL columns onto final_df
            final_df = pd.merge(
                final_df,
                pose_skl_merged,
                on=['system', 'itemid'],
                how='left'
            )
            print("\n=== Final DataFrame with Pose SKL Metrics ===")
            print(final_df)
        else:
            print("No valid pose SKL data found to merge.")
    else:
        print("No pose SKL files found matching pattern 'pose_skl*.csv'.")

    # ------------------------------
    # STEP 5c: Join SignCLIP metrics
    # ------------------------------
    signclip_files = glob.glob(os.path.join(script_dir, "metrics.*.sign_clip.csv"))
    if signclip_files:
        sc_dfs = []
        for sc_file in signclip_files:
            try:
                df_sc = pd.read_csv(sc_file).rename(columns={"example_id": "itemid"})
                # Keep only metric columns, exclude source/target/etc.
                cols = ["system", "itemid"] + [
                    c for c in df_sc.columns
                    if c not in {"data", "example_id", "system", "itemid", "source", "target"}
                ]
                sc_dfs.append(df_sc[cols])
            except Exception as e:
                print(f"Error reading SignCLIP file '{sc_file}': {e}")
        if sc_dfs:
            signclip_df = pd.concat(sc_dfs, ignore_index=True)
            signclip_df = signclip_df.loc[:, ~signclip_df.columns.duplicated()]
            final_df = pd.merge(
                final_df,
                signclip_df,
                on=["system", "itemid"],
                how="left"
            )
            print("\n=== Final DataFrame with SignCLIP Metrics ===")
            print(final_df)
        else:
            print("No valid SignCLIP data found to merge.")
    else:
        print("No SignCLIP files found matching pattern 'metrics.*.sign_clip.csv'.")

    # ------------------------------
    # STEP 5b: Join Pose EvalTop10 metrics (new fourth source)
    # ------------------------------
    pose_eval_files = glob.glob(os.path.join(script_dir, "metrics.*.pose_evaltop10.csv"))
    if pose_eval_files:
        pe_dfs = []
        for pe_file in pose_eval_files:
            try:
                df_pe = pd.read_csv(pe_file).rename(columns={"example_id": "itemid"})
                # drop the raw 'data' column, then select system, itemid, plus only the actual metric columns
                cols = ["system", "itemid"] + [
                    c for c in df_pe.columns
                    if c not in {"data", "example_id", "system", "itemid"}
                ]
                pe_dfs.append(df_pe[cols])
            except Exception as e:
                print(f"Error reading Pose EvalTop10 file '{pe_file}': {e}")
        if pe_dfs:
            pose_eval_df = pd.concat(pe_dfs, ignore_index=True)
            # sanity check: drop any accidental duplicates
            pose_eval_df = pose_eval_df.loc[:, ~pose_eval_df.columns.duplicated()]

            final_df = pd.merge(
                final_df,
                pose_eval_df,
                on=["system", "itemid"],
                how="left"
            )
            print("\n=== Final DataFrame with Pose EvalTop10 Metrics ===")
            print(final_df)
        else:
            print("No valid Pose EvalTop10 data found to merge.")
    else:
        print("No Pose EvalTop10 files found matching pattern 'metrics.*.pose_evaltop10.csv'.")

    # ------------------------------
    # STEP 6: Check for missing values
    # ------------------------------
    if final_df.isnull().values.any():
        print("\n=== Error: Missing values detected in the final DataFrame! ===")
        print(final_df.isnull().sum())
    else:
        print("\n=== Final DataFrame Check: No missing values found! ===")
    
    # ------------------------------
    # STEP 7: Aggregate rows by system
    # ------------------------------
    aggregated_system_df = final_df.groupby('system')[
        ['score'] + [col for col in final_df.columns if col not in ['system', 'itemid', 'score', 'direction']]
    ].mean()
    print("\n=== Aggregated DataFrame by System (Average of each Column/Metric) ===")
    print(aggregated_system_df)

    # ------------------------------
    # STEP 8: Aggregate rows by system and direction
    # ------------------------------
    aggregated_system_direction_df = final_df.groupby(['system', 'direction'])[
        ['score'] + [col for col in final_df.columns if col not in ['system','itemid','score','direction']]
    ].mean()
    print("\n=== Aggregated DataFrame by System and Direction (Average of each Column/Metric) ===")
    print(aggregated_system_direction_df)

    # ------------------------------
    # STEP 8.5: Split off the 'ref' system
    # ------------------------------
    non_ref_df = final_df[final_df['system'] != 'ref']
    ref_df     = final_df[final_df['system'] == 'ref']

    # ------------------------------
    # STEP 9: Correlation Analysis on non-ref systems
    # ------------------------------
    # make an explicit copy so we can safely mutate
    df_corr = non_ref_df.copy()

    # list of core error‐style metrics we want to flip
    error_metrics = [
        'nMSE', 'nAPE', 'DTW', 'nDTW',
        'SkeletonVAE', 'SkeletonVAE_path_norm', 'SkeletonVAE_ref_norm',
        'SKL_mvt_hshp_h2p', 'SKL_mvt_hshp_wo_norm_h2p',
        'SKL_mvt_h2p', 'SKL_mvt_wo_norm_h2p'
    ]

    # flip the sign of each named metric …
    for col in error_metrics:
        if col in df_corr.columns:
            df_corr.loc[:, col] = -df_corr.loc[:, col]

    # …and also flip any column containing the substring "DTWAggregatedDistanceMetricFast"
    for col in df_corr.columns:
        if "DTWAggregatedDistanceMetricFast" in col:
            df_corr.loc[:, col] = -df_corr.loc[:, col]

    # recompute aggregates on df_corr
    aggregated_system_df = df_corr.groupby('system')[
        ['score'] + [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]
    ].mean()
    aggregated_system_direction_df = df_corr.groupby(['system','direction'])[
        ['score'] + [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]
    ].mean()

    metric_columns = [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]

    # Overall segment‐level correlation
    overall_seg_corr_pearson = df_corr[['score'] + metric_columns].corr(method='pearson').loc['score', metric_columns]
    overall_seg_corr_spearman = df_corr[['score'] + metric_columns].corr(method='spearman').loc['score', metric_columns]

    # Segment‐level by system
    seg_corrs_pearson_system = {}
    seg_corrs_spearman_system = {}
    for sys in df_corr['system'].unique():
        sys_df = df_corr[df_corr['system'] == sys]
        sys_numeric = sys_df[['score'] + metric_columns]
        seg_corrs_pearson_system[sys] = sys_numeric.corr(method='pearson').loc['score', metric_columns]
        seg_corrs_spearman_system[sys] = sys_numeric.corr(method='spearman').loc['score', metric_columns]

    # Segment‐level by direction
    seg_corrs_pearson_direction = {}
    seg_corrs_spearman_direction = {}
    for d in df_corr['direction'].unique():
        dir_df = df_corr[df_corr['direction'] == d]
        dir_numeric = dir_df[['score'] + metric_columns]
        seg_corrs_pearson_direction[d] = dir_numeric.corr(method='pearson').loc['score', metric_columns]
        seg_corrs_spearman_direction[d] = dir_numeric.corr(method='spearman').loc['score', metric_columns]

    # --- NEW: compute counts for each category ---
    seg_counts = {}
    seg_counts['Overall'] = len(df_corr)
    for sys in df_corr['system'].unique():
        seg_counts[sys] = len(df_corr[df_corr['system'] == sys])
    for d in df_corr['direction'].unique():
        seg_counts[d] = len(df_corr[df_corr['direction'] == d])

    # --- Build and annotate the Pearson table ---
    seg_corrs_pearson = {
        "Overall": overall_seg_corr_pearson,
        **seg_corrs_pearson_system,
        **seg_corrs_pearson_direction
    }
    seg_corrs_pearson_df = pd.DataFrame(seg_corrs_pearson).T

    # insert #seg as first column
    seg_corrs_pearson_df.insert(
        0,
        '#seg',
        [seg_counts.get(idx, np.nan) for idx in seg_corrs_pearson_df.index]
    )
    # compute SD over just the metric columns, leave #seg as NaN
    pearson_stds = seg_corrs_pearson_df.drop(columns=['#seg']).std()
    seg_corrs_pearson_df.loc['SD'] = [np.nan] + pearson_stds.tolist()

    print("\n=== NON-REF: Segment-level Pearson Correlation with Human Score ===")
    print(seg_corrs_pearson_df)


    # --- Build and annotate the Spearman table ---
    seg_corrs_spearman = {
        "Overall": overall_seg_corr_spearman,
        **seg_corrs_spearman_system,
        **seg_corrs_spearman_direction
    }
    seg_corrs_spearman_df = pd.DataFrame(seg_corrs_spearman).T

    # insert #seg as first column
    seg_corrs_spearman_df.insert(
        0,
        '#seg',
        [seg_counts.get(idx, np.nan) for idx in seg_corrs_spearman_df.index]
    )
    # compute SD over just the metric columns, leave #seg as NaN
    spearman_stds = seg_corrs_spearman_df.drop(columns=['#seg']).std()
    seg_corrs_spearman_df.loc['SD'] = [np.nan] + spearman_stds.tolist()

    print("\n=== NON-REF: Segment-level Spearman Correlation with Human Score ===")
    print(seg_corrs_spearman_df)

    # System‐level correlation (Overall)
    system_corr_pearson = aggregated_system_df.corr(method='pearson').loc['score', aggregated_system_df.columns != 'score']
    system_corr_spearman = aggregated_system_df.corr(method='spearman').loc['score', aggregated_system_df.columns != 'score']

    # System‐level correlation by direction
    system_corrs_pearson_direction = {}
    system_corrs_spearman_direction = {}
    for d in df_corr['direction'].unique():
        dir_df = df_corr[df_corr['direction'] == d]
        dir_numeric = dir_df[['score'] + metric_columns]
        system_corrs_pearson_direction[d] = dir_numeric.corr(method='pearson').loc['score', dir_numeric.columns != 'score']
        system_corrs_spearman_direction[d] = dir_numeric.corr(method='spearman').loc['score', dir_numeric.columns != 'score']

    system_corrs_pearson = {
        "Overall": system_corr_pearson,
        **system_corrs_pearson_direction
    }
    system_corrs_spearman = {
        "Overall": system_corr_spearman,
        **system_corrs_spearman_direction
    }

    system_corr_pearson_df = pd.DataFrame(system_corrs_pearson).T
    system_corr_spearman_df = pd.DataFrame(system_corrs_spearman).T

    system_corr_pearson_df.loc['SD'] = system_corr_pearson_df.std()
    print("\n=== NON-REF: System-level Pearson Correlation with Human Score ===")
    print(system_corr_pearson_df)

    # after system_corr_spearman_df = pd.DataFrame(...)
    system_corr_spearman_df.loc['SD'] = system_corr_spearman_df.std()
    print("\n=== NON-REF: System-level Spearman Correlation with Human Score ===")
    print(system_corr_spearman_df)

    # after printing segment-level Pearson
    pearson_seg_out = os.path.join(script_dir, "correlation_v2.segment.pearson.csv")
    seg_corrs_pearson_df.to_csv(pearson_seg_out, index=True)
    print(f"\n=== Saved Pearson segment-level correlations to '{pearson_seg_out}' ===")

    # after printing segment-level Spearman
    spearman_seg_out = os.path.join(script_dir, "correlation_v2.segment.spearman.csv")
    seg_corrs_spearman_df.to_csv(spearman_seg_out, index=True)
    print(f"\n=== Saved Spearman segment-level correlations to '{spearman_seg_out}' ===")

    # after printing system-level Pearson
    pearson_sys_out = os.path.join(script_dir, "correlation_v2.system.pearson.csv")
    system_corr_pearson_df.to_csv(pearson_sys_out, index=True)
    print(f"\n=== Saved Pearson system-level correlations to '{pearson_sys_out}' ===")

    # after printing system-level Spearman
    spearman_sys_out = os.path.join(script_dir, "correlation_v2.system.spearman.csv")
    system_corr_spearman_df.to_csv(spearman_sys_out, index=True)
    print(f"\n=== Saved Spearman system-level correlations to '{spearman_sys_out}' ===")

    # ------------------------------
    # STEP 10: Correlation Analysis on ref system
    # ------------------------------
    if not ref_df.empty:
        df_corr = ref_df

        # recompute aggregates on df_corr
        aggregated_system_df = df_corr.groupby('system')[
            ['score'] + [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]
        ].mean()
        aggregated_system_direction_df = df_corr.groupby(['system','direction'])[
            ['score'] + [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]
        ].mean()

        metric_columns = [col for col in df_corr.columns if col not in ['system','itemid','score','direction']]

        # Overall segment‐level correlation
        overall_seg_corr_pearson = df_corr[['score'] + metric_columns].corr(method='pearson').loc['score', metric_columns]
        overall_seg_corr_spearman = df_corr[['score'] + metric_columns].corr(method='spearman').loc['score', metric_columns]

        # Segment‐level by direction (only direction-level, since there's a single system)
        seg_corrs_pearson_direction = {
            d: df_corr[df_corr['direction'] == d][['score'] + metric_columns].corr(method='pearson').loc['score', metric_columns]
            for d in df_corr['direction'].unique()
        }
        seg_corrs_spearman_direction = {
            d: df_corr[df_corr['direction'] == d][['score'] + metric_columns].corr(method='spearman').loc['score', metric_columns]
            for d in df_corr['direction'].unique()
        }

        # Combine for ref
        seg_corrs_pearson = {"Overall": overall_seg_corr_pearson, **seg_corrs_pearson_direction}
        seg_corrs_spearman = {"Overall": overall_seg_corr_spearman, **seg_corrs_spearman_direction}

        seg_corrs_pearson_df = pd.DataFrame(seg_corrs_pearson).T
        seg_corrs_spearman_df = pd.DataFrame(seg_corrs_spearman).T

        print("\n=== REF: Segment-level Pearson Correlation with Human Score ===")
        print(seg_corrs_pearson_df)
        print("\n=== REF: Segment-level Spearman Correlation with Human Score ===")
        print(seg_corrs_spearman_df)

        # System‐level (overall and by direction)
        system_corr_pearson = aggregated_system_df.corr(method='pearson').loc['score', aggregated_system_df.columns != 'score']
        system_corr_spearman = aggregated_system_df.corr(method='spearman').loc['score', aggregated_system_df.columns != 'score']

        system_corrs_pearson_direction = {}
        system_corrs_spearman_direction = {}
        for d in df_corr['direction'].unique():
            dir_df = df_corr[df_corr['direction'] == d]
            dir_numeric = dir_df[['score'] + metric_columns]
            system_corrs_pearson_direction[d] = dir_numeric.corr(method='pearson').loc['score', dir_numeric.columns != 'score']
            system_corrs_spearman_direction[d] = dir_numeric.corr(method='spearman').loc['score', dir_numeric.columns != 'score']

        system_corrs_pearson = {"Overall": system_corr_pearson, **system_corrs_pearson_direction}
        system_corrs_spearman = {"Overall": system_corr_spearman, **system_corrs_spearman_direction}

        system_corr_pearson_df = pd.DataFrame(system_corrs_pearson).T
        system_corr_spearman_df = pd.DataFrame(system_corrs_spearman).T

        print("\n=== REF: System-level Pearson Correlation with Human Score ===")
        print(system_corr_pearson_df)
        print("\n=== REF: System-level Spearman Correlation with Human Score ===")
        print(system_corr_spearman_df)
    else:
        print("No data found for system 'ref'; skipping STEP 10.")

if __name__ == "__main__":
    main()
