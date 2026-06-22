import os
import pandas as pd
from sklearn.model_selection import train_test_split # For stratified sampling
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def create_stratified_sample(df, sample_size=12000):
    print(f"Creating a stratified sample of {sample_size} records...")
    # Perform stratified split based on product_category
    strat_df, _ = train_test_split(
        df, 
        train_size=sample_size, 
        stratify=df['product_category'], 
        random_state=42
    )
    return strat_df

def chunk_and_index(df_sample, db_path="vector_store/"):
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Task specifications: 500 characters chunk size, 50 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # Initialize Persistent ChromaDB Client
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="local_complaints_sample")
    
    documents = []
    metadatas = []
    ids = []
    
    print("Processing and chunking narratives...")
    for idx, row in df_sample.iterrows():
        chunks = text_splitter.split_text(row['cleaned_narrative'])
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{row.get('complaint_id', idx)}_{i}"
            
            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append({
                "complaint_id": str(row.get('complaint_id', idx)),
                "product_category": row['product_category'],
                "issue": str(row.get('issue', 'Unknown')),
                "company": str(row.get('company', 'Unknown'))
            })
            
    print(f"Generated {len(documents)} total text chunks. Generating embeddings and adding to ChromaDB...")
    
    # Batch insertion to optimize speed
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch_docs = documents[i:end_idx]
        batch_ids = ids[i:end_idx]
        batch_meta = metadatas[i:end_idx]
        
        # Compute embeddings manually via SentenceTransformer
        batch_embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Indexed chunks {i} to {end_idx} successfully.")

if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/filtered_complaints.csv"
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH)
        sample_df = create_stratified_sample(df, sample_size=12000)
        chunk_and_index(sample_df)
    else:
        print("Processed data not found. Please run Task 1 preprocessing pipeline first.")