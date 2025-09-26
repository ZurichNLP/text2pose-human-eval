import os
import csv
import argparse
import pandas as pd
import numpy as np
import math

# Prevent output truncation
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# Set these constants as appropriate.
LIKERT_SCALE = 100        # Maximum continuous score (adjust as needed)
DISCRETIZED_SCALE = 6     # Target discrete scale: 0–6

# ---------------- Fleiss' Kappa Functions ----------------

def fleiss_kappa(M, strict=False):
    """
    Compute Fleiss' kappa given a rating matrix M.
    Each row corresponds to a subject (e.g. an item) and each column is a rating category
    (the number of ratings in that category).
    If strict is True, only rows where all annotators rated are considered.
    Returns a string formatted as "kappa±stdErr".
    """
    M = np.array(M)
    n_annotators = float(max(M.sum(axis=1)))
    if strict:
        M = M[np.argwhere(np.sum(M, axis=1) == n_annotators)].squeeze(1)
    N, k = M.shape  # N = number of subjects, k = number of categories
    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    P_bar = np.sum(P) / N
    P_barE = np.sum(p * p)
    if 1 - P_barE == 0:
        return None
    kappa = (P_bar - P_barE) / (1 - P_barE)
    stdErr = math.sqrt(P_bar * (1 - P_bar) / (N * (1 - P_barE) ** 2))
    return f"{kappa:.2f}±{stdErr:.2f}"

def compute_intra_agreement(user_df, scale=DISCRETIZED_SCALE, score_key='score'):
    """
    Compute intra-annotator agreement (Fleiss' κ) for a given annotator's DataFrame (user_df)
    based on repeated ratings for the same (system, itemid).
    Continuous scores are first scaled from LIKERT_SCALE to 0–scale, rounded, and clamped.
    Returns a string "kappa±std (N)" where N is the number of items (rows in the rating matrix),
    if repeated items exist; otherwise, returns None.
    """
    df = user_df.copy()
    factor = LIKERT_SCALE / scale
    df['disc_score'] = df[score_key].apply(lambda x: int(round(x / factor)) if pd.notnull(x) else x)
    df['disc_score'] = df['disc_score'].apply(lambda x: max(0, min(x, scale)) if pd.notnull(x) else x)
    df['uniqID'] = df['system'].astype(str) + "_" + df['itemid'].astype(str)
    uniqIDs = df['uniqID'].unique()
    rating_matrix = []
    for uid in uniqIDs:
        subset = df[df['uniqID'] == uid]
        if len(subset) > 1:
            counts = [ (subset['disc_score'] == r).sum() for r in range(scale+1) ]
            rating_matrix.append(counts)
    if len(rating_matrix) == 0:
        return None
    kappa_str = fleiss_kappa(rating_matrix, strict=True)
    if kappa_str is None:
        return None
    return f"{kappa_str} ({len(rating_matrix)})"

def compute_inter_agreement(df, scale=DISCRETIZED_SCALE, score_key='score', 
                            exclude_annotators=[], group_by='system', min_evaluation_per_user=100):
    """
    Compute inter-annotator agreement (Fleiss' κ) for a given system or for the entire aggregated DataFrame.
    First, exclude annotations from annotators in exclude_annotators.
    Then, discretize continuous scores (scale from LIKERT_SCALE to 0–scale).
    For each item (grouped by itemid), group ratings by annotator and average multiple ratings,
    round the result, and build a rating matrix using items rated by at least two annotators.
    If group_by is 'system', the agreement is computed for each system.
    If group_by is 'overall', the agreement is computed across all systems.
    Returns a string "kappa±std (N)" if repeated items exist; otherwise, returns None.
    """
    df = df.copy()
    df = df[~df['username'].isin(exclude_annotators)]
    factor = LIKERT_SCALE / scale
    df['disc_score'] = df[score_key].apply(lambda x: int(round(x / factor)) if pd.notnull(x) else x)
    df['disc_score'] = df['disc_score'].apply(lambda x: max(0, min(x, scale)) if pd.notnull(x) else x)
    
    # Filter users based on the number of evaluations
    user_counts = df['username'].value_counts()
    valid_users = user_counts[user_counts >= min_evaluation_per_user].index
    df = df[df['username'].isin(valid_users)]
    
    # Adjust grouping based on the 'group_by' parameter
    if group_by == 'system':
        group_field = 'system'
    elif group_by == 'overall':
        group_field = 'uniqID'
        df['uniqID'] = df['system'].astype(str) + "_" + df['itemid'].astype(str)
    else:
        raise ValueError("Invalid value for 'group_by'. Use 'system' or 'overall'.")
    
    rating_matrix = []
    for group_value, group in df.groupby(group_field):
        if group_by == 'system':
            # For system-level agreement, group by itemid and then by annotator
            for itemid, item_group in group.groupby('itemid'):
                user_ratings = item_group.groupby('username')['disc_score'].mean().round().astype(int)
                if len(user_ratings) > 1:
                    counts = [(user_ratings == r).sum() for r in range(scale+1)]
                    rating_matrix.append(counts)
        elif group_by == 'overall':
            # For overall-level agreement, group by unique itemid
            user_ratings = group.groupby('username')['disc_score'].mean().round().astype(int)
            if len(user_ratings) > 1:
                counts = [(user_ratings == r).sum() for r in range(scale+1)]
                rating_matrix.append(counts)
    
    if len(rating_matrix) == 0:
        return None
    
    kappa_str = fleiss_kappa(rating_matrix, strict=True)
    if kappa_str is None:
        return None
    return f"{kappa_str} ({len(rating_matrix)})"

# ---------------- Main Processing Functions ----------------

def read_header(header_path):
    """Read the column names from the header CSV file."""
    try:
        with open(header_path, newline='') as header_file:
            header_reader = csv.reader(header_file)
            column_names = next(header_reader)
        return column_names
    except Exception as e:
        print(f"Error reading header file {header_path}: {e}")
        return None

def adjust_itemid(row):
    """
    Adjust the itemid based on the system.
    For ambiguous systems, subtract the appropriate base:
      - For 'sockeye': subtract 30000 if 30000 ≤ itemid < 31000, or 31000 if 31000 ≤ itemid < 40000.
      - For 'sign_mt_v2': subtract 40000 if 40000 ≤ itemid < 41000, or 41000 if itemid ≥ 41000.
      - For 'ref': subtract 20000 if itemid ≥ 20000.
      - Otherwise, leave itemid unchanged.
    """
    itemid = row['itemid']
    system = row['system']
    if pd.isnull(itemid):
        return itemid
    if system == "sockeye":
        if 30000 <= itemid < 31000:
            return itemid - 30000
        elif 31000 <= itemid < 40000:
            return itemid - 31000
        else:
            return itemid
    elif system == "sign_mt_v2":
        if 40000 <= itemid < 41000:
            return itemid - 40000
        elif itemid >= 41000:
            return itemid - 41000
        else:
            return itemid
    elif system == "ref":
        if itemid >= 20000:
            return itemid - 20000
        else:
            return itemid
    else:
        return itemid

def process_csv_files(input_dir, account_dir, header_path, csv_files, output_file, highlight_user):
    """
    Process each CSV file:
      - Read the CSV using the header.
      - Merge with account CSV (if available) to replace 'username' with 'participant'.
      - Filter rows where isdocumentlevelscore is False.
      - Keep only the columns: username, system, itemid, itemtype, srclang, trglang, score.
      - Exclude rows where username == "Anne".
      - For round 2 files, adjust itemid by subtracting the appropriate base.
      - Sort rows by ['username', 'system', 'itemid'].
    Then, aggregate all processed DataFrames, sort the final aggregated DataFrame by the same keys,
    convert itemid to int, save to output_file, compute merged statistics (including intra- and inter-annotator agreement),
    and if highlight_user is provided, log that user's annotations to a file.
    """
    column_names = read_header(header_path)
    if column_names is None:
        return

    aggregated_dfs = []
    total_original_rows = 0

    for file_name in csv_files:
        full_input_path = os.path.join(input_dir, file_name)
        if not os.path.exists(full_input_path):
            print(f"File not found: {full_input_path}")
            continue
        try:
            human_df = pd.read_csv(full_input_path, header=None, names=column_names)
            original_count = human_df.shape[0]
        except Exception as e:
            print(f"Error reading {full_input_path}: {e}")
            continue
        total_original_rows += original_count

        full_account_path = os.path.join(account_dir, file_name)
        if os.path.exists(full_account_path):
            try:
                account_df = pd.read_csv(full_account_path)
                merged_df = pd.merge(
                    human_df,
                    account_df[['Username', 'participant']],
                    left_on='username',
                    right_on='Username',
                    how='left'
                )
                merged_df['username'] = merged_df['participant']
                merged_df.drop(['participant', 'Username'], axis=1, inplace=True)
                human_df = merged_df
            except Exception as e:
                print(f"Error merging account data for {file_name}: {e}")
        else:
            print(f"Account file not found for {file_name} at {full_account_path}. Skipping merge for this file.")

        if 'isdocumentlevelscore' in human_df.columns:
            human_df = human_df[human_df['isdocumentlevelscore'] == False]
        else:
            print(f"Warning: 'isdocumentlevelscore' column not found in {file_name}. No filtering applied.")

        desired_columns = ['username', 'system', 'itemid', 'itemtype', 'srclang', 'trglang', 'score']
        missing_columns = [col for col in desired_columns if col not in human_df.columns]
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in {file_name}. They will be ignored.")
            desired_columns = [col for col in desired_columns if col in human_df.columns]
        human_df = human_df[desired_columns]
        human_df = human_df[human_df['username'] != "Anne"]

        if "r2" in file_name:  # Check if the file is from round 2
            human_df['itemid'] = pd.to_numeric(human_df['itemid'], errors='coerce')
            human_df['itemid'] = human_df.apply(adjust_itemid, axis=1)

        human_df = human_df.sort_values(by=['username', 'system', 'itemid'])
        print(f"Contents of {file_name}:")
        print(human_df)
        print("-" * 80)
        aggregated_dfs.append(human_df)

    if aggregated_dfs:
        aggregated_df = pd.concat(aggregated_dfs, ignore_index=True)
        aggregated_df = aggregated_df.sort_values(by=['username', 'system', 'itemid'])
        final_count = aggregated_df.shape[0]
        aggregated_df['itemid'] = aggregated_df['itemid'].astype('Int64')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        aggregated_df.to_csv(output_file, index=False)
        print(f"Aggregated CSV file saved to {output_file}")

        total_processed_rows = sum(df.shape[0] for df in aggregated_dfs)
        if total_processed_rows != final_count:
            print(f"Sanity check FAILED: Sum of processed rows = {total_processed_rows}, but final aggregated rows = {final_count}")
        else:
            print(f"Sanity check PASSED: Total aggregated rows = {final_count}")

        aggregated_df['score'] = pd.to_numeric(aggregated_df['score'], errors='coerce')
        pd.set_option('display.max_colwidth', None)

        # --- Per User Evaluation Statistics ---
        user_stats = aggregated_df.groupby('username')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
        user_system_stats = aggregated_df.groupby(['username', 'system'])['score'].agg(['mean', 'count']).reset_index()
        pivot = user_system_stats.pivot(index='username', columns='system')
        desired_systems = ['ref', 'sockeye', 'sign_mt', 'sign_mt_v2']
        mean_count_df = pd.DataFrame(index=pivot.index)

        def format_mean_count(m, c):
            if pd.isna(m) or pd.isna(c):
                return "N/A"
            else:
                return f"{m:.2f}/{int(c)}"

        for system in desired_systems:
            pivot_mean = pivot['mean'].reindex(columns=[system])
            pivot_count = pivot['count'].reindex(columns=[system])
            if system in pivot_mean.columns and system in pivot_count.columns:
                mean_count_df[f"mean/count_{system}"] = pivot_mean[system].combine(
                    pivot_count[system],
                    lambda m, c: format_mean_count(m, c)
                )
            else:
                mean_count_df[f"mean/count_{system}"] = "N/A"

        user_stats = user_stats.join(mean_count_df)

        # --- Compute intra-agreement for each user ---
        intra_agreement = {}
        for user in aggregated_df['username'].unique():
            user_df = aggregated_df[aggregated_df['username'] == user]
            kappa = compute_intra_agreement(user_df, scale=DISCRETIZED_SCALE, score_key='score')
            intra_agreement[user] = kappa if kappa is not None else "N/A"

        user_stats['intra_agreement (#rows)'] = user_stats.index.map(intra_agreement)

        # --- Calculate mean intra-agreement over users with valid intra-agreement values ---
        valid_intra_agreement = [kappa for kappa in intra_agreement.values() if kappa != "N/A"]
        if valid_intra_agreement:
            mean_intra_agreement = np.mean([float(kappa.split('±')[0]) for kappa in valid_intra_agreement])
        else:
            mean_intra_agreement = "N/A"

        # Add the mean intra-agreement as an additional row
        user_stats.loc['Mean Intra-Agreement'] = user_stats.mean(numeric_only=True)
        user_stats.loc['Mean Intra-Agreement', 'intra_agreement (#rows)'] = f"{mean_intra_agreement:.2f}" if isinstance(mean_intra_agreement, float) else mean_intra_agreement

        print("\n--- Per User Evaluation Statistics ---")
        print(user_stats)

        # --- compute average stderr of per-user κ ---
        stderr_vals = []
        for kappa in intra_agreement.values():
            if kappa != "N/A":
                # e.g. "0.62±0.11 (50)" → split off the "(50)", then split on '±'
                main = kappa.split()[0]        # "0.62±0.11"
                stderr = main.split('±')[1]    # "0.11"
                stderr_vals.append(float(stderr))

        if stderr_vals:
            avg_stderr = np.mean(stderr_vals)
            print(f"\nAverage intra-agreement stderr (over users): {avg_stderr:.2f}")
        else:
            print("\nNo intra-agreement stderr values to average.")


        # --- Per System Evaluation Statistics ---
        system_stats = aggregated_df.groupby('system')['score'].agg(['count', 'mean', 'std', 'min', 'max'])
        system_users = aggregated_df.groupby('system')['username'].unique().apply(lambda x: ", ".join(x))
        system_stats = system_stats.join(system_users.rename("evaluated_users"))
        desired_systems_order = ['ref', 'sockeye', 'sign_mt', 'sign_mt_v2']
        system_stats = system_stats.reindex(desired_systems_order)
        inter_agreement = {}
        for system in desired_systems_order:
            sys_df = aggregated_df[aggregated_df['system'] == system]
            kappa = compute_inter_agreement(sys_df, scale=DISCRETIZED_SCALE, score_key='score', 
                                            exclude_annotators=['Lisa0', 'Lisa'])
            inter_agreement[system] = kappa if kappa is not None else "N/A"
        system_stats['inter_agreement (#rows)'] = pd.Series(inter_agreement)
        overall_inter = compute_inter_agreement(aggregated_df, scale=DISCRETIZED_SCALE, score_key='score',
                                                        exclude_annotators=['Lisa0', 'Lisa'], group_by='overall')
        overall_stats = pd.Series({
            'count': aggregated_df['score'].count(),
            'mean': aggregated_df['score'].mean(),
            'std': aggregated_df['score'].std(),
            'min': aggregated_df['score'].min(),
            'max': aggregated_df['score'].max(),
            'evaluated_users': ", ".join(aggregated_df['username'].unique()),
            'inter_agreement (#rows)': overall_inter if overall_inter is not None else "N/A"
        }, name="Total")
        system_stats = pd.concat([system_stats, overall_stats.to_frame().T])
        print("\n--- Per System Evaluation Statistics (sorted by fixed system order, with Total row) ---")
        print(system_stats)

        # --- Highlight User Annotations ---
        if highlight_user is not None:
            highlight_df = aggregated_df[aggregated_df['username'] == highlight_user]
            if not highlight_df.empty:
                highlight_file = os.path.join(os.path.dirname(output_file), f"aggregated_{highlight_user}.csv")
                highlight_df.to_csv(highlight_file, index=False)
                print(f"\nAnnotations for highlighted user '{highlight_user}' have been logged to: {highlight_file}")
            else:
                print(f"\nNo annotations found for highlighted user '{highlight_user}'.")
    else:
        print("No CSV files were processed successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="CLI program to process human annotation CSV files, merge with account data, adjust itemid, filter out testing user, and aggregate results with merged statistics, a sanity check, and intra-/inter-annotator agreement."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="human_evaluation/batches_text2pose/results/",
        help="Input directory containing human annotation CSV files."
    )
    parser.add_argument(
        "--account-dir",
        type=str,
        default="human_evaluation/batches_text2pose/accounts",
        help="Directory containing account CSV files."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="human_evaluation/batches_text2pose/results/aggregated.csv",
        help="Output CSV file path."
    )
    parser.add_argument(
        "--header-file",
        type=str,
        default="human_evaluation/scores/header/header.csv",
        help="Path to the header CSV file."
    )
    parser.add_argument(
        "--highlight-user",
        type=str,
        default=None,
        help="Optional: if set, log all rows for this user to a file instead of printing to stdout."
    )
    args = parser.parse_args()

    csv_files = [
        "text2poseSignsuisse.deu-sgg.ref.csv",
        "text2poseSignsuisse.deu-sgg.sockeye.csv",
        "text2poseSignsuisse.deu-sgg.sign_mt.csv",
        "text2poseSignsuisse.deu-sgg.r2.dsgs03.csv",
        "text2poseSignsuisse.deu-sgg.r2.csv",
        "text2poseSignsuisse.fra-fsl.r2.csv",  # Add the French annotation file
        "text2poseSignsuisse.ita-ise.r2.csv",  # Add the Italian annotation file
    ]

    process_csv_files(args.input_dir, args.account_dir, args.header_file, csv_files, args.output_file, args.highlight_user)

if __name__ == "__main__":
    main()