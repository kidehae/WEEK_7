import gradio as gr
from src.rag_engine import CrediTrustRAG

# Initialize your engine
print("Starting Gradio Interface Application Engine...")
try:
    rag = CrediTrustRAG()
except Exception as e:
    print(f"Engine still downloading or waiting: {e}")
    rag = None

def chatbot_interface(question):
    if rag is None:
        return "RAG Engine is still initializing. Please wait until the model finishes downloading in the terminal.", "No sources available."
    
    try:
        # Call the core processing backend from Task 3
        answer, sources = rag.generate_response(question)
        
        # Build a beautiful, clear citation layout for Asha and compliance auditing
        source_text = ""
        for i, src in enumerate(sources):
            source_text += f"### Source Reference {i+1}\n"
            source_text += f"**Product Pillar:** {src['metadata'].get('product_category', 'N/A')} | "
            source_text += f"**Issue Filed:** {src['metadata'].get('issue', 'N/A')} | "
            source_text += f"**Company:** {src['metadata'].get('company', 'N/A')}\n\n"
            source_text += f"> *\"{src['text']}\"*\n\n"
            source_text += "---\n\n"
            
        return answer, source_text
    except Exception as e:
        return f"An processing error occurred: {str(e)}", "No sources extracted."

# Building the Interactive UI Custom Canvas
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust Financial — Intelligent Complaint Analysis Portal")
    gr.Markdown("Transform raw, unstructured mobile-first customer feedback narratives into strategic, evidence-backed actions immediately.")
    
    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="Ask Asha's Analytical Question", 
                placeholder="e.g., Why are people unhappy with Credit Cards?",
                lines=3
            )
            submit_btn = gr.Button("Analyze Complaints Pipeline", variant="primary")
            clear_btn = gr.Button("Clear Workspace")
        
        with gr.Column(scale=3):
            output_answer = gr.Markdown(label="Synthesized Strategic Insight")
            
    gr.Markdown("## 📑 Grounded Evidence & Verification Sources")
    output_sources = gr.Markdown(label="Retrieved Narrative Contexts")
    
    # Map interactive behaviors
    submit_btn.click(
        fn=chatbot_interface,
        inputs=[user_input],
        outputs=[output_answer, output_sources]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[user_input, output_answer, output_sources]
    )

if __name__ == "__main__":
    demo.launch(share=False)