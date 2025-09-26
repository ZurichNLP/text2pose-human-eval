import os
import csv
import json
import matplotlib.pyplot as plt

import pandas as pd


# Set pandas to display all columns and avoid truncation of column content
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_colwidth', None)  # Display the full content of each column
pd.set_option('display.width', None)  # Prevent line breaking for wide outputs


csv_header_filename = 'human_evaluation/scores/header/header.csv'
with open(csv_header_filename) as header_file:
    header_reader = csv.reader(header_file)
    column_names = next(header_reader)

with open('./metrics/metrics_back_translation.json', 'r') as f:
    bt_data = json.load(f)
with open('./metrics/metrics_signmt_v2.json', 'r') as f:
    bt_data_2 = json.load(f)
bt_data.update(bt_data_2)

vae_path = './metrics/metrics_vae_l2_mean.csv'
vae_df = pd.read_csv(vae_path)
print('vae_df:', vae_df.describe())

correlation_dfs = []
metrics_all = []
human_scores_all = []
systems = ['ref', 'sockeye', 'sign_mt', 'sign_mt_v2']
for system in systems:
    human_path = f'./human_evaluation/batches_text2pose/results/text2poseSignsuisse.deu-sgg.{system}.csv'
    account_path = f'./human_evaluation/batches_text2pose/accounts/text2poseSignsuisse.deu-sgg.{system}.csv'

    # FIXME: not yet human annotation and VAE scores for sign.mt v2
    if system == 'sign_mt_v2': 
        human_path = f'./human_evaluation/batches_text2pose/results/text2poseSignsuisse.deu-sgg.sign_mt.csv'
        account_path = f'./human_evaluation/batches_text2pose/accounts/text2poseSignsuisse.deu-sgg.sign_mt.csv'
        vae_df['sign_mt_v2'] = vae_df['sign_mt']

    human_df = pd.read_csv(human_path, names=column_names)
    account_df = pd.read_csv(account_path)

    account_df = account_df[account_df['participant'] != 'Lisa0']
    # if system == 'sign_mt':
    account_df = account_df[account_df['participant'] != 'Lisa']
    account_df = account_df[account_df['participant'] != 'dsgs01']

    # Merge the two dataframes on the condition where 'username' in human_df equals 'Username' in account_df
    merged_df = pd.merge(human_df, account_df[['Username', 'participant']], left_on='username', right_on='Username', how='left')
    # Now, replace 'username' column in human_df with 'participant' from account_df
    merged_df['username'] = merged_df['participant']
    # Drop the extra 'participant' and 'Username' columns from merged_df
    merged_df.drop(['participant', 'Username'], axis=1, inplace=True)
    human_df = merged_df
    
    # Pivot the dataframe to get each username as a column
    pivot_df = human_df.pivot_table(index='itemid', columns='username', values='score', aggfunc='first')
    # Rename the columns to start with 'score_'
    pivot_df.columns = ['score_' + col for col in pivot_df.columns]
    # Add a new column for average score for each itemid
    pivot_df['score_avg'] = pivot_df.mean(axis=1)
    # Reset the index to make itemid a column again
    pivot_df.reset_index(inplace=True)
    human_df = pivot_df

    # only select the rows where every annotators have annotated
    human_df.dropna(inplace=True)
   
    human_scores = [col for col in human_df.columns if col.startswith('score_')]

    if system != 'ref':
        metrics_path = f'./metrics/metrics.{system}.csv'
        metrics_df = pd.read_csv(metrics_path)

        # merge back-translated metrics from Biao/Google
        bt_df = pd.DataFrame(bt_data[system])
        bt_df.rename(columns={'score': 'likelihood'}, inplace=True)
        bt_df['chrf'] = bt_df['chrf'].str.extract(r'[\=]\s*(\d+\.\d+)')
        bt_df['chrf'] = bt_df['chrf'].astype(float)
        bt_df['bleu'] = bt_df['bleu'].str.extract(r'BLEU = (\d+\.\d+)')
        bt_df['bleu'] = bt_df['bleu'].astype(float)
        bt_df['bleurt'] = bt_df['bleurt'] * 100
        metrics_df = pd.concat([metrics_df, bt_df], axis=1)

        # if system == 'sign_mt_v2':
        #     print(bt_df[:10])
        #     print(metrics_df[:10])
        #     exit()

        # merge SkeletonVAE scores
        # Merge the DataFrames on 'example_id' from metrics_df and 'sentence_id' from vae_df
        merged_df = pd.merge(metrics_df, vae_df[['sentence_id', system]], left_on='example_id', right_on='sentence_id', how='left')
        # Drop the 'sentence_id' column if it's no longer needed
        merged_df.drop(columns=['sentence_id'], inplace=True)
        merged_df.rename(columns={system: 'vae_l2_mean'}, inplace=True)
        metrics_df = merged_df

        # # Loop through all CSV files in the folder
        # folder_path = './metrics/skeleton_vae'
        # vae_metrics = []
        # for file_name in os.listdir(folder_path):
        #     # Check if the file matches the naming pattern
        #     if file_name.endswith('.csv') and 'L2_across_time_averaged_over_dims_' in file_name:
        #         # Extract the number 'xx' from the filename
        #         file_number = file_name.split('_')[6]  # Assuming the filename structure is consistent
                
        #         # Construct the full path to the CSV file
        #         file_path = os.path.join(folder_path, file_name)

        #         # Read the CSV into a DataFrame
        #         vae_df = pd.read_csv(file_path)

        #         # Check the structure of the vae_df (optional)
        #         print(f"Processing file: {file_name}")
        #         # print(vae_df.describe())

        #         # Merge the current CSV DataFrame (vae_df) with metrics_df
        #         merged_df = pd.merge(metrics_df, vae_df[['sentence_id', system]], 
        #                             left_on='example_id', right_on='sentence_id', how='left')

        #         # Drop the 'sentence_id' column if it's no longer needed
        #         merged_df.drop(columns=['sentence_id'], inplace=True)

        #         # Rename the column from 'vae_l2_mean' to 'vae_xx_radius' based on the number extracted from the filename
        #         merged_df.rename(columns={system: f'vae_{file_number}_radius'}, inplace=True)

        #         # Update metrics_df with the merged data
        #         metrics_df = merged_df

        #         vae_metrics.append(f'vae_{file_number}_radius')
        # print(metrics_df)

        # merge human scores
        human_df['example_id'] = human_df['itemid']
        selected_columns = ['example_id'] + [col for col in human_df.columns if col.startswith('score_')]
        correlation_df = metrics_df.merge(human_df[selected_columns], on='example_id', how='inner')

        # metrics = ['nMSE', 'nAPE', 'DTW', 'nDTW', 'SegDiff', 'SegnAPERecall']
        metrics = ['score_avg', 'nMSE', 'nAPE', 'DTW', 'nDTW', 'bleu', 'chrf', 'bleurt', 'likelihood', 'vae_l2_mean']
        # metrics = metrics + vae_metrics 

        correlation_dfs.append({
            # 'human_scores': correlation_df[human_scores],
            'metrics': correlation_df[metrics],
        })
        human_scores_all.append(correlation_df[human_scores])
        metrics_all.append(correlation_df[metrics])
    else:
         # filter document-level scores and non-score columns
        human_df = human_df[human_df['itemid'] < 1000000]

        correlation_dfs.append({
            # 'human_scores': human_df[human_scores],
            'metrics': None,
        })
        human_scores_all.append(human_df[human_scores])

aggregated_df = {
    # 'human_scores': pd.concat(human_scores_all),
    'metrics': pd.concat(metrics_all),
}
correlation_dfs.append(aggregated_df)

for i, correlation_df_dict in enumerate(correlation_dfs):
    system = systems[i] if i < len(systems) else 'aggregated'
    print(f'System: {system}')

    for score, correlation_df in correlation_df_dict.items():
        if correlation_df is not None:
            print(score)

            # if score == 'metrics':
            #     # Min-Max normalized
            #     metrics = ['nMSE', 'nAPE', 'DTW', 'nDTW'] 
            #     correlation_df[metrics] = 100 * (correlation_df[metrics] - correlation_df[metrics].min()) / (correlation_df[metrics].max() - correlation_df[metrics].min())

            # pd.set_option('display.max_rows', 1000)
            print(correlation_df)
            print(correlation_df.describe())
            print()

            for method in ['pearson', 'spearman']:
            # for method in ['pearson', 'spearman', 'kendall']:
                print(f'{method} correlation')

                corr = correlation_df.corr(method=method)
                print(corr)

                if score == 'human_scores':
                    print(f'mean {method} correlation with others')
                    print(corr.mean())
                    print()

                # plt.matshow(corr)
                # plt.savefig(f'./metrics/figures/correlation.{system}.{method}.png', bbox_inches='tight')

    print('===============================')
