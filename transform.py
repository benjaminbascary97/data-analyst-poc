import pandas as pd
import csv
def safe_convert_delimiter(input_file_path, output_file_path, original_delimiter=';', new_delimiter=','):
    # Load the original CSV with the correct delimiter
    df = pd.read_csv(input_file_path, sep=original_delimiter)
    
    # Save the DataFrame to a new CSV file with the desired delimiter and quoting all fields
    df.to_csv(output_file_path, index=False, sep=new_delimiter, quoting=csv.QUOTE_ALL)

# Specify the paths to your original and new files
input_file_path = 'output.csv'
output_file_path = 'output2.csv'

# Convert the file delimiters from semicolons to commas safely
safe_convert_delimiter(input_file_path, output_file_path)