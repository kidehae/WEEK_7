import os
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

# def load_data(file_path):
#     print(f"Loading data from {file_path}...")
#     # Adjust loader depending on your file extension (.csv, .parquet etc.)
#     if file_path.endswith('.parquet'):
#         return pd.read_parquet(file_path)
#     return pd.read_csv(file_path, low_memory=False)


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)
    
    # Clean up column names: replace spaces/hyphens with underscores and lowercase them
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    print("Available columns after normalization:", list(df.columns))
    return df

def run_eda(df):
    print("\n--- Running EDA ---")
    # 1. Total complaints vs complaints with narratives
    total_records = len(df)
    has_narrative = df['consumer_complaint_narrative'].notna().sum()
    print(f"Total Records: {total_records}")
    print(f"Records with narratives: {has_narrative} ({has_narrative/total_records*100:.2f}%)")
    
    # 2. Focus only on the 4 targeted categories
    # Note: Map exactly to how they appear in your dataset (e.g., 'Credit card or prepaid card')
    target_products = ['Credit card', 'Credit card or prepaid card', 'Personal loan', 'Payday loan, title loan, or personal loan', 'Savings account', 'Checking or savings account', 'Money transfer, virtual currency, or money service']
    
    # Simple verification mapping for exact matches if standard CFPB names apply:
    print("\nInitial Product Distribution:")
    print(df['product'].value_counts().head(10))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove common CFPB boilerplate masks like XXXX
    text = re.sub(r'x{2,}', '', text)
    # Remove special characters and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_pipeline(input_path, output_path):
    df = load_data(input_path)
    run_eda(df)
    
    # Filter for non-empty narratives
    df = df[df['consumer_complaint_narrative'].notna() & (df['consumer_complaint_narrative'] != "")]
    
    # Standardize and filter by the 4 required product pillars
    # Modify these strings based on your specific raw dataset variant definitions
    product_mapping = {
        'Credit card': 'Credit Card',
        'Credit card or prepaid card': 'Credit Card',
        'Personal loan': 'Personal Loan',
        'Payday loan, title loan, or personal loan': 'Personal Loan',
        'Savings account': 'Savings Account',
        'Checking or savings account': 'Savings Account',
        'Money transfer': 'Money Transfer',
        'Money transfer, virtual currency, or money service': 'Money Transfer'
    }
    
    df['product_category'] = df['product'].map(product_mapping)
    df = df[df['product_category'].notna()]
    
    print(f"\nFiltered Dataset Count by Category:\n{df['product_category'].value_counts()}")
    
    # Apply text cleaning
    print("\nCleaning text narratives...")
    df['cleaned_narrative'] = df['consumer_complaint_narrative'].apply(clean_text)
    
    # Filter out empty narratives post-cleaning
    df = df[df['cleaned_narrative'] != ""]
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved clean dataset to {output_path}")

if __name__ == "__main__":
    # Point this to where your raw dataset is stored
    RAW_DATA_PATH = "data/raw/complaints.csv" 
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    
    # Execute if the raw file exists
    if os.path.exists(RAW_DATA_PATH):
        preprocess_pipeline(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    else:
        print(f"Please place your raw CFPB dataset at {RAW_DATA_PATH} to execute.")