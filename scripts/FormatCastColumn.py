# This script formats the 'Star Cast' column in the IMDb dataset to ensure proper name formatting.
# It handles cases with initials, first and last names, and removes any unwanted commas.


import pandas as pd
import re

# Load the CSV file
df = pd.read_csv('IMDb_Dataset_Composite.csv')

def format_star_cast(cast_string):
    if pd.isna(cast_string):
        return cast_string
    # Replace any existing commas to avoid conflicts
    cast_string = cast_string.replace(',', '')
    # Split names based on uppercase letters, preserving initials like J.R.R.
    names = re.findall(r'[A-Z][a-z]*(?:\.[A-Z]\.)*(?=[A-Z]|\s|$)', cast_string)
    # Join names with commas, handling spaces and initials
    formatted_names = []
    i = 0
    while i < len(names):
        name = names[i]
        # Check for initials like J.R.R.
        if '.' in name:
            formatted_names.append(name)
            i += 1
        else:
            # Pair first and last names
            if i + 1 < len(names) and '.' not in names[i + 1]:
                formatted_names.append(f"{name} {names[i + 1]}")
                i += 2
            else:
                formatted_names.append(name)
                i += 1
    return ', '.join(formatted_names)

# Apply the formatting to the Star Cast column
df['Star Cast'] = df['Star Cast'].apply(format_star_cast)

# Save the updated DataFrame to a new CSV
df.to_csv('IMDb_Dataset_Composite_Formatted.csv', index=False)
print("Formatted CSV saved as 'IMDb_Dataset_Composite_Formatted.csv'")