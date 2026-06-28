# YMCA Staff Chatbot

A polished internal staff assistant built with Retrieval-Augmented Generation (RAG) to answer questions from YMCA handbook materials using Anthropic Claude, LlamaIndex, ChromaDB, and Streamlit.

## What this project does

This application helps YMCA staff find answers from internal and public handbook sources without relying on guesswork. It ingests:

- the employee handbook PDF stored in the docs/ folder
- the public member handbook from https://www.ymcala.org/handbook/

The chatbot then retrieves the most relevant chunks from those documents and uses Claude to answer questions in a grounded, document-based way.

## Tech stack

- Python
- Streamlit for the chat interface
- LlamaIndex for document loading and retrieval
- ChromaDB for vector storage and similarity search
- Anthropic Claude API for answer generation
- Pytest for automated testing

## Project structure

- app.py — Streamlit frontend
- chatbot.py — ingestion, retrieval, and answer logic
- ingest.py — rebuilds the knowledge base from the current sources
- tests/test_chatbot.py — regression tests for ingestion and retrieval
- docs/ — handbook source files

## Local setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file with your Anthropic API key:
   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   ```
4. Rebuild the knowledge base:
   ```bash
   python ingest.py
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Updating documents and hours

When policies, procedures, or branch hours change:

1. Update the source file in docs/ or replace the scraped website source
2. Re-run the ingestion pipeline:
   ```bash
   python ingest.py
   ```
3. Refresh the Streamlit app to use the updated knowledge base

This ensures new policies are added and outdated ones are replaced as the source materials evolve.

## Testing

Run the test suite with:

```bash
pytest
```
