# Medical RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot for medical documents.

This project:
- Loads medical PDF files from the `data/` folder
- Splits documents into chunks
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in a local FAISS index
- Uses `google/flan-t5-base` to answer user questions from retrieved context
- Provides a chat UI with Streamlit

## Project Structure

```text
medical_rag_bot/
├── app.py                          # Streamlit UI + RAG chat flow
├── main.py                         # Optional entry script (if used)
├── pyproject.toml                  # Project dependencies
├── README.md
├── data/                           # Put your PDF files here
├── src/
│   ├── memory_for_llm.py           # Builds FAISS index from PDFs
│   └── connect_memory_to_llm.py    # CLI RAG flow (terminal input)
└── vector_store/
	└── faiss_index/
		└── index.faiss             # Saved FAISS index
```

## Requirements

- Python 3.12+
- Local machine with enough RAM for model loading
- Internet connection on first run to download model files (unless already cached)

## Install

### Option 1: With `uv` (recommended)

```bash
uv sync
```

### Option 2: With `pip`

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -e .
```

## Add Your Medical PDFs

Place one or more PDF files in:

```text
data/
```

## Build / Refresh the Vector Database

Run:

```bash
python src/memory_for_llm.py
```

This script will:
- Read PDFs from `data/`
- Create chunks
- Generate embeddings
- Save FAISS index to `vector_store/faiss_index`

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## How the RAG Flow Works

1. User asks a question in Streamlit chat.
2. FAISS retriever fetches the most relevant chunks from your indexed PDFs.
3. A context-grounded prompt is created.
4. FLAN-T5 generates an answer using only retrieved context.
5. The app displays answer + source snippets.

## Configuration

Current defaults in the code:
- LLM: `google/flan-t5-base`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB path: `vector_store/faiss_index`
- Retrieval top-k: configurable in Streamlit sidebar

## Troubleshooting

### 1) Empty or poor answers
- Ensure PDFs are present in `data/`.
- Rebuild FAISS index with `python src/memory_for_llm.py`.
- Increase retrieved chunks (`k`) from sidebar.

### 2) Model download or load issues
- Check internet for first-time model pull.
- Confirm enough disk/RAM.
- Re-run after model cache completes.

### 3) FAISS loading error
- Verify files exist under `vector_store/faiss_index/`.
- Recreate index by rerunning indexing script.

### 4) Slow first response
- First query is slower due to model load.
- Later responses are faster because resources are cached.

## Notes

- This is an educational local RAG setup and not a certified medical device.
- Do not use this system as a replacement for professional medical advice.

## License

Add your preferred license here.
