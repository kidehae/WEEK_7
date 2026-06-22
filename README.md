

# Intelligent Complaint Analysis for Financial Services (RAG Chatbot)

An internal, production-grade AI platform built for **CrediTrust Financial** to convert unstructured, raw customer complaint narratives into actionable product insights. Utilizing a Retrieval-Augmented Generation (RAG) architecture, this tool enables product managers, support leaders, and compliance teams to query over 464K customer narratives across four major financial lines using natural English.

---

## ── Business Value & KPIs

* **Accelerate Discovery:** Reduces the time required for a Product Manager to pinpoint a major complaint trend from days to seconds.
* **Democratize Data:** Empowers non-technical staff (Support/Compliance) to extract insights without writing SQL or waiting on a data analyst.
* **Proactive Posture:** Shifts operations from reactive troubleshooting to real-time, evidence-based feedback analysis.

---

## ── Project Architecture & Repository Structure

The layout follows the exact specifications outlined in the project challenge document:

```text
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml          # Automated CI pipeline for test suites
├── data/
│   ├── raw/                       # Raw CFPB data dropzone
│   └── processed/                 # Standardized, cleaned and filtered CSVs
├── vector_store/                  # Persistent ChromaDB/FAISS indexes
├── notebooks/
│   ├── __init__.py
│   └── README.md                  # EDA and prototyping records
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Text cleaning & extraction pipeline
│   ├── indexing.py                # Stratification, chunking & vector injection
│   └── rag_engine.py              # Retrieval & LLM prompt orchestration
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py           # PyTest test cases
├── app.py                         # User interface (Gradio application)
├── requirements.txt               # Main project dependencies
├── README.md                      # Documentation
└── .gitignore                     # Data & cache guardrails

```

---

## ── Core Technical Stack

| Pipeline Layer | Component Chosen | Justification |
| --- | --- | --- |
| **Data Orchestration** | `Pandas` & `Scikit-Learn` | For low-memory processing and exact stratified splitting. |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunk size: 500, Overlap: 50. Preserves contextual sentences. |
| **Embedding Engine** | `sentence-transformers/all-MiniLM-L6-v2` | Low memory footprint (~80MB), 384 dimensions, high semantic accuracy. |
| **Vector DB** | `ChromaDB` (Persistent) | Native metadata filtering and seamless local storage integration. |
| **Generator (LLM)** | Open-Source LLM Pipeline | Grounded exclusively using context-injected engineering prompts. |
| **User Interface** | `Gradio` | Simple implementation supporting live citation streams and clean rendering. |

---

## ── Getting Started

### 1. Environment Configuration & Installation

Clone the repository, initialize a Python 3.12 environment, and install the required dependencies:

```bash
git clone https://github.com/your-username/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### 2. Run Task 1: Preprocessing Pipeline

Place your raw CFPB dataset at `data/raw/complaints.csv` and execute the pipeline from the project root:

```bash
python src/preprocess.py

```

This filters for the 4 targeted financial pillars (**Credit Card, Personal Loan, Savings Account, Money Transfer**), cleans out boilerplate phrases and `XXXX` anonymization masks, and exports to `data/processed/filtered_complaints.csv`.

### 3. Run Task 2: Sample Embedding & Vector Ingestion

To build and verify your vector database locally using a stratified validation sample of 12,000 narratives:

```bash
python src/indexing.py

```

This partitions your data proportionally, splits it into roughly **33,338 text chunks**, embeds them, and saves the database to the `vector_store/` directory.

### 4. Run Tasks 3 & 4: Launching the App

To start the Gradio UI using the full pre-built database index:

```bash
python app.py

```

Open the provided local URL (e.g., `http://127.0.0.1:7860`) in your browser to interact with the system.

---

## ── Evaluation Metrics

Our system relies on an exact prompt structure that instructs the generation engine to act as a focused financial analyst assistant. The engine is constrained to use *only* the retrieved context blocks, outputting clear source citations underneath every answer to ensure auditing teams can verify facts immediately.
