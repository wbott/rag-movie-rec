import pandas as pd
import re

def add_delimiters(name_string):
    # Protect names with initials (e.g., "J.R.R. Tolkien") by replacing with a temporary placeholder
    initial_names = {}
    def protect_initials(match):
        key = f"__INITIAL_{len(initial_names)}__"
        initial_names[key] = match.group(0)
        return key
    protected_string = re.sub(r'[A-Z]\.[A-Z]\.[A-Z]\.\s*\w+', protect_initials, name_string)
    
    # Handle camel case transitions (e.g., "JohnLogan" -> "John,Logan")
    split_names = re.sub(r'([a-z])([A-Z])', r'\1,\2', protected_string)
    # Replace multiple spaces with a single comma (e.g., "Shameik Moore" -> "Shameik,Moore")
    split_names = re.sub(r'\s+', ',', split_names.strip())
    # Restore spaces within names (e.g., "Brian,Tyree,Henry" -> "Brian Tyree,Henry")
    split_names = re.sub(r'(\w+),(\w+)(,|$)', r'\1 \2\3', split_names)
    
    # Restore protected initial names (e.g., "__INITIAL_0__" -> "J.R.R. Tolkien")
    for key, value in initial_names.items():
        split_names = split_names.replace(key, value)
    
    # Ensure no double commas
    split_names = re.sub(r',+', ',', split_names)
    return split_names

def process_csv(input_file, output_file, column_name):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Apply the delimiter function to the specified column
    df[column_name] = df[column_name].apply(add_delimiters)
    
    # Save the modified DataFrame to a new CSV
    df.to_csv(output_file, index=False)
    print(f"Processed CSV saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_csv = "IMDb_Dataset_Composite.csv"
    output_csv = "IMDb_Dataset_Composite_Cleaned.csv"
    actor_column = "Star Cast"
    process_csv(input_csv, output_csv, actor_column)