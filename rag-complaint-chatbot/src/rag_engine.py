import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class CrediTrustRAG:
    def __init__(self, db_path="vector_store/", collection_name="local_complaints_sample"):
        print("Initializing Embedding Model (all-MiniLM-L6-v2)...")
        self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print(f"Connecting to Vector DB at {db_path}...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        print("Loading Generator LLM Pipeline...")
        # Switched to 'text-generation' task with TinyLlama for local compatibility
        self.generator = pipeline(
            "text-generation", 
            model="MBZUAI/LaMini-GPT-124M", 
            max_new_tokens=256,
            temperature=0.3
        )
    def retrieve(self, query, k=5):
        """Step 1 & 2: Embed query and perform similarity search"""
        query_vector = self.embed_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=k
        )
        return results

    def generate_response(self, question):
        """Step 3: Combine prompt context and query into Generator LLM"""
        retrieved_data = self.retrieve(question, k=3)
        
        chunks = retrieved_data['documents'][0]
        metadatas = retrieved_data['metadatas'][0]
        
        # Build the structured context block
        context_str = "\n".join([f"- [Source: {meta['product_category']}]: {text}" for text, meta in zip(chunks, metadatas)])
        
        # Exact Prompt Template required by 10 Academy challenge document
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions
about customer complaints. Use the following retrieved complaint excerpts to formulate
your answer. If the context doesn't contain the answer, state that you don't have
enough information.

Context: {context_str}

Question: {question}

Answer:"""

        # Generate text using the pipeline
       # Generate text using the causal text-generation pipeline
        gen_output = self.generator(prompt)
        full_text = gen_output[0]['generated_text']
        
        # Cleanly extract just the newly generated text after our prompt template
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text
        
        # Package answer alongside sources for verification rendering in Task 4
        sources = [{"text": txt, "metadata": meta} for txt, meta in zip(chunks, metadatas)]
        return answer, sources

if __name__ == "__main__":
    # Quick internal validation test
    rag_system = CrediTrustRAG()
    test_query = "Why are people unhappy with Credit Cards?"
    ans, src = rag_system.generate_response(test_query)
    
    print("\n=== SAMPLE QUERY TEST ===")
    print(f"Question: {test_query}")
    print(f"Answer: {ans}")