# Setup database for testing
import kagglehub
import os
import pandas as pd

kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')

if not os.path.isfile(kaggle_json_path):
    print(f"""Kaggle API key not found!
    
To use the Kaggle API, you must have your credentials saved at:
    {kaggle_json_path}. Otherwise this script will quietly fail.

Please download your kaggle.json from your Kaggle account settings:
    https://www.kaggle.com/settings/account

Then place it at the above location.
""")
    sys.exit()  
else:
    print(f"Found Kaggle API credentials at: {kaggle_json_path}")

# Download the dataset
path = kagglehub.dataset_download("parthdande/imdb-dataset-2024-updated")
print("Path to dataset files:", path)

# List of input CSV filenames
input_files = ['IMDb_Dataset.csv', 'IMDb_Dataset_2.csv', 'IMDb_Dataset_3.csv']

# Output file - delete if exists
output_file= os.path.join(os.getcwd(), 'IMDb_Dataset_Composite.csv')
if os.path.isfile(output_file):
    os.remove(output_file)

composite_df = pd.DataFrame()
replaced_titles = []  # Track titles that got replaced

for file in input_files:
    file_path = os.path.join(path, file)
    if os.path.exists(file_path):

        df = pd.read_csv(file_path,encoding='utf-8')

        # Drop duplicate columns
        df = df.drop_duplicates(subset='Title', keep='last')

        if composite_df.empty:
            composite_df = df.copy()
        else:

            # Set Title as index for easier updating
            df = df.set_index('Title')
            composite_df = composite_df.set_index('Title')

            # Find which Titles will be replaced
            common_titles = df.index.intersection(composite_df.index)
            replaced_titles.extend(common_titles.tolist())

            # Update existing Titles
            composite_df.update(df)

            # Add new Titles
            composite_df = pd.concat([composite_df, df[~df.index.isin(composite_df.index)]])

            # Reset index back to normal
            composite_df = composite_df.reset_index()

        print(f"Merged file: {file}")
    else:
        print(f"Warning: {file_path} not found. Skipping.")

# Handle null values
composite_df['Poster-src'] = composite_df['Poster-src'].fillna('No Poster Available')
composite_df['Second_Genre'] = composite_df['Second_Genre'].fillna('')
composite_df['Third_Genre'] = composite_df['Third_Genre'].fillna('')

# Rollup genres into a single column
def combine_genres(row):
    genres = [row['Genre'], row['Second_Genre'], row['Third_Genre']]
    # Remove any None, NaN, or empty string values
    genres = [g for g in genres if pd.notnull(g) and str(g).strip() != '']
    return ", ".join(genres)

composite_df['Genres'] = composite_df.apply(combine_genres, axis=1)
composite_df = composite_df.drop(columns=['Genre', 'Second_Genre', 'Third_Genre'])

# Save the final composite file
output_file = 'IMDb_Dataset_Composite.csv'
composite_df.to_csv(output_file, index=False)
print(f"Composite dataset saved as {output_file}")

print("\n=======> Null values per column:")
print(composite_df.isnull().sum())

duplicates = composite_df.duplicated().sum()
print(f"\n=======> Number of duplicate rows: {duplicates}")
