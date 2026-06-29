# YMCA Staff Chatbot

A polished RAG-powered chatbot for YMCA staff that answers questions from handbook documents using Anthropic Claude, LlamaIndex, ChromaDB, and Streamlit.

## Features

- Ingests the local employee handbook PDF and a public member handbook source
- Stores and retrieves document chunks in ChromaDB
- Provides a clean Streamlit chat interface for staff questions
- Supports policy updates by refreshing the knowledge base from the latest source documents
- Includes a test suite for ingestion and answer behavior

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```
4. Place handbook files in the docs/ directory.
5. Run the ingestion script:
   ```bash
   python ingest.py
   ```
6. Start the app:
   ```bash
   streamlit run app.py
   ```

## Updating policies and hours

When branch hours or policies change, update the source documents in the docs/ folder or the website source, then rerun the ingestion step so the new information replaces the old content in the knowledge base.

## Testing

Run the tests with:

```bash
pytest
```
