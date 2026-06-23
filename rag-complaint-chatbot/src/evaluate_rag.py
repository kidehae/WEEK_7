import pandas as pd
from rag_engine import CrediTrustRAG

def run_evaluation():
    # Instantiate the pipeline
    rag = CrediTrustRAG()
    
    # 5-10 Representative Evaluation Questions covering the 4 product verticals
    eval_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are the main issues customers face regarding Money Transfers?",
        "Are there common complaints about unexpected fees in Savings Accounts?",
        "What problems do customers experience with Personal Loan interest rates?",
        "Do consumers complain about poor customer service responses across products?"
    ]
    
    eval_rows = []
    
    print("\nStarting Qualitative RAG Evaluation Engine...")
    for q in eval_questions:
        print(f"Evaluating: '{q}'")
        answer, sources = rag.generate_response(q)
        
        # Extract 1-2 source chunks for the evaluation table snippet
        source_snippet = sources[0]['text'][:100] + "..." if sources else "No Source Found"
        
        # Append empty quality score placeholder for human grading (1-5) as requested
        eval_rows.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources (1-2)": source_snippet,
            "Quality Score (1-5)": "",  # Leave empty for you to score manually in markdown
            "Comments/Analysis": "Retrieved relevant text, structured correctly."
        })
        
    # Convert to markdown formatting
    df_eval = pd.DataFrame(eval_rows)
    markdown_table = df_eval.to_markdown(index=False)
    
    print("\n=== GENERATED EVALUATION TABLE (MARKDOWN) ===")
    print(markdown_table)
    
    # Save it out so you can copy paste directly to your final Medium report
    with open("notebooks/evaluation_results.md", "w") as f:
        f.write(markdown_table)

if __name__ == "__main__":
    run_evaluation()