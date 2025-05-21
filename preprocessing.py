import pandas as pd

# Load the dataset from the 'Data' directory
df = pd.read_csv("Data/aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Remove rows where the language is German ('de')
df = df[df['language'] != 'de']

# Rename the 'queue' column to 'Sub-Category'
df.rename(columns={'queue': 'Sub-Category'}, inplace=True)

# Function to assign high-level categories based on sub-categories
def assign_category(sub_category):
    if sub_category in [
        'Technical Support', 'Product Support', 'Returns and Exchanges',
        'Billing and Payments', 'Sales and Pre-Sales', 'General Inquiry'
    ]:
        return 'Customer Support'
    elif sub_category in ['IT Support', 'Service Outages and Maintenance']:
        return 'IT & Infrastructure'
    elif sub_category in ['Human Resources', 'Customer Service']:
        return 'Internal Operations'
    else:
        return 'Unknown'  
# Apply the category assignment function
df['Category'] = df['Sub-Category'].apply(assign_category)

# Drop unnecessary columns from the dataset
columns_to_drop = [
    'answer', 'type', 'priority', 'language', 'version',
    'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8'
]
df = df.drop(columns=columns_to_drop, axis=1)

# Save the cleaned and preprocessed dataset to a new CSV file
df.to_csv("Data/preprocessed_data.csv", index=False)
