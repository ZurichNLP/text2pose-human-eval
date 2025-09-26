import pandas as pd
import argparse
import os

def extract_unique_ids(csv_file, output_txt_file, excluded_ids):
    """Extract unique IDs from the third column of the CSV file, filter out specified IDs, and write them to a sorted text file."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter out rows where the first column is in the excluded_ids list
    df_filtered = df[~df.iloc[:, 0].isin(excluded_ids)]
    
    # Assuming the ID is in the third column (index 2, zero-indexed)
    ids = df_filtered.iloc[:, 2].unique()  # Extract unique values from the third column
    ids = [id for id in ids if id < 10000]
    
    # Sort the IDs
    sorted_ids = sorted(ids)
    
    # Write the sorted unique IDs to the output text file
    with open(output_txt_file, 'w') as f:
        for _id in sorted_ids:
            f.write(f"{_id}\n")
    
    print(f"Unique IDs have been written to {output_txt_file}")

def process_folder(input_folder):
    """Process all CSV files in the specified folder and generate .ids.txt and .ids.all.txt files."""
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}.")
        return
    
    # IDs to be excluded for the first file (excludes "deusgg0304", "deusgg0204", "deusgg0201")
    exclude_first = ["deusgg0304", "deusgg0204", "deusgg0201"]
    # IDs to be excluded for the second file (only excludes "deusgg0201")
    exclude_second = ["deusgg0201"]
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_file_path = os.path.join(input_folder, csv_file)
        
        # Generate the output text file paths
        output_txt_file = csv_file_path.replace('.csv', '.id.txt')
        output_all_txt_file = csv_file_path.replace('.csv', '.id.all.txt')
        
        # Extract unique IDs and write to the output files
        extract_unique_ids(csv_file_path, output_txt_file, exclude_first)
        extract_unique_ids(csv_file_path, output_all_txt_file, exclude_second)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract unique IDs from CSV files in a folder, filter out specific IDs, and write to sorted text files.")
    parser.add_argument("input_folder", type=str, nargs='?', default="./human_evaluation/batches_text2pose/results/", help="The path to the input folder containing CSV files (default: './human_evaluation/batches_text2pose/results/')")
    args = parser.parse_args()
    
    # Get the input folder path
    input_folder = args.input_folder
    
    # Check if the folder exists
    if not os.path.isdir(input_folder):
        print(f"Error: The folder {input_folder} does not exist.")
        return
    
    # Process the folder and generate .ids.txt and .ids.all.txt files for all CSV files
    process_folder(input_folder)

if __name__ == "__main__":
    main()
