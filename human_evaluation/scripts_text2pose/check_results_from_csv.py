import pandas as pd
import argparse
import os

def count_rows_and_item_ids_by_user_and_system(csv_file):
    """Count the number of rows for each user and system, and collect unique item ids."""
    # Read the CSV file
    df = pd.read_csv(csv_file, header=None, names=["user", "system", "item_id", "type", "source", "target", "score", "text", "is_complete", "start_time", "end_time"])
    
    # Group by 'user' and 'system', and count the rows
    count = df.groupby(['user', 'system']).size().reset_index(name='row_count')
    
    # For each user-system pair, get the unique item ids
    item_ids = df.groupby(['user', 'system'])['item_id'].unique().reset_index(name='unique_item_ids')
    
    # Merge the row count and unique item ids dataframes
    result = pd.merge(count, item_ids, on=['user', 'system'])
    
    # Group by user to get the total rows and total unique item ids
    user_summary = df.groupby('user').agg(
        total_rows=('item_id', 'size'),
        total_unique_item_ids=('item_id', pd.Series.nunique)
    ).reset_index()

    # Print user-wise summary first
    for _, summary in user_summary.iterrows():
        print(f"User: {summary['user']}")
        print(f"  Total rows: {summary['total_rows']}")
        print(f"  Total unique item IDs: {summary['total_unique_item_ids']}")
        print("-" * 40)
    
    # Print details for each user-system pair
    for _, row in result.iterrows():
        print(f"User: {row['user']}")
        print(f"  System: {row['system']}")
        print(f"    Row count: {row['row_count']}")
        print(f"    Unique item IDs ({len(row['unique_item_ids'])}): {', '.join(map(str, sorted(row['unique_item_ids'])))}")
        print("-" * 40)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Count rows for each user and system combination in the given CSV file and list unique item IDs.")
    parser.add_argument("csv_file", type=str, nargs='?', default="./human_evaluation/batches_text2pose/results/text2poseSignsuisse.deu-sgg.r2.dev.csv", help="Path to the CSV file")
    args = parser.parse_args()

    # Get the CSV file path
    csv_file = args.csv_file
    
    # Check if the file exists
    if not os.path.isfile(csv_file):
        print(f"Error: The file {csv_file} does not exist.")
        return
    
    # Process the CSV file and print the counts and unique item IDs
    count_rows_and_item_ids_by_user_and_system(csv_file)

if __name__ == "__main__":
    main()
