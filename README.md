# AquiLLM


![AquiLLM Logo](aquillm/aquillm/static/images/aquila.svg)

[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.1-green.svg)](https://www.djangoproject.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**AquiLLM is an open-source RAG (Retrieval-Augmented Generation) application designed specifically for researchers.** It helps you manage, search, and interact with your research documents using AI, streamlining your literature review and knowledge discovery process. Upload various document formats, organize them into collections, and chat with an AI that understands your library's content.

More info can be found at [[https://aquillm.org]]

<!-- ![AquiLLM Screenshot](path/to/screenshot.gif) -->

## Key Features

*   **Versatile Document Ingestion**: Upload PDFs, fetch arXiv papers by ID, import VTT transcripts, scrape webpages, and process handwritten notes (with OCR).
*   **Intelligent Organization**: Group documents into logical `Collections` for focused research projects.
*   **AI-Powered Chat**: Engage in context-aware conversations with your documents, ask follow-up questions, and get answers with source references.

## Tech Stack

*   **Backend**: Python, Django
*   **Frontend**: React
*   **Database**: PostgreSQL
*   **Vector Store**: pgvector (PostgreSQL extension)
*   **LLM Integration**: Claude, OpenAI, Gemini, or local LLMs (via Ollama)
*   **Asynchronous Tasks**: Celery, Redis, Django Channels

*   **Authentication**: django-allauth
*   **Containerization**: Docker, Docker Compose

## Quick start (development and local use):

This assumes you have Docker and Docker Compose installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AquiLLM/AquiLLM.git 
    cd AquiLLM
    ```
2.  **Copy the environment template:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the .env file with your specific configuration:**
    - **Database settings**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_NAME`, `POSTGRES_HOST`
    - **LLM backend** (set **exactly one** of the following, see also “Configuring the LLM backend” below):
      - Hosted APIs:
        - `LLM_CHOICE=CLAUDE` &rarr; requires `ANTHROPIC_API_KEY`
        - `LLM_CHOICE=OPENAI` &rarr; requires `OPENAI_API_KEY`
        - `LLM_CHOICE=GEMINI` &rarr; requires `GEMINI_API_KEY`
      - Local / OSS via Ollama (no external API key required):
        - `LLM_CHOICE=GEMMA3` (uses `ebdm/gemma3-enhanced:12b` through Ollama)
        - `LLM_CHOICE=LLAMA3.2`
        - `LLM_CHOICE=GPT-OSS:120b` (via the `gpt-oss:120b` Ollama model)
      - To switch models after initial setup, update `LLM_CHOICE` in `.env` and do a full restart:
        ```bash
        docker compose down && docker compose up
        ```
        A simple restart may not pick up the change.

4.  **Build and run using Docker Compose:**
    ```bash
    docker compose up -d
    ```

4. **Add a superuser:**
   ```bash
   docker compose exec web ./manage.py addsuperuser
   ```

5.  **Access the application:**

    Open your browser to `http://localhost:8080`, sign in with superuser account.

7.  **Stop the application:**
    ```bash
    docker compose down
    ```

## Updating (development):

Pull the latest changes and rebuild. Migrations run automatically on startup.

```bash
git pull origin main
docker compose down && docker compose up --build -d
```

## Updating (production):

```bash
git pull origin main
docker compose -f docker-compose-prod.yml down && docker compose -f docker-compose-prod.yml up --build -d
```

## Small-scale deployment:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AquiLLM/AquiLLM.git 
    cd AquiLLM
    ```
2.  **Copy the environment template:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the .env file with your specific configuration:**
    - **Database settings**: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_NAME`, `POSTGRES_HOST`
    - **LLM backend**:
      - Hosted APIs:
        - `LLM_CHOICE=CLAUDE` &rarr; requires `ANTHROPIC_API_KEY`
        - `LLM_CHOICE=OPENAI` &rarr; requires `OPENAI_API_KEY`
        - `LLM_CHOICE=GEMINI` &rarr; requires `GEMINI_API_KEY`
      - Local / OSS via Ollama:
        - `LLM_CHOICE=GEMMA3`
        - `LLM_CHOICE=LLAMA3.2`
        - `LLM_CHOICE=GPT-OSS`
      - To switch models after initial setup, update `LLM_CHOICE` in `.env` and do a full restart:
        ```bash
        docker compose down && docker compose -f docker-compose-prod.yml up --build -d
        ```
    - Optional: Google OAuth credentials (GOOGLE_OAUTH2_CLIENT_ID, GOOGLE_OAUTH2_CLIENT_SECRET)
    - Optional: Email access permissions (ALLOWED_EMAIL_DOMAINS, ALLOWED_EMAIL_ADDRESSES). Required if OAuth is to be used.
    - Set HOST_NAME for your domain or use 'localhost' for development

4.  **Build and run using Docker Compose:**
    ```bash
    docker compose -f docker-compose-prod.yml up  -d 
    ```
    This will automatically use letsencrypt to get TLS certificates.

4. **Add a superuser for administration:**
   ```bash
   docker compose -f docker-compose-prod.yml exec web ./manage.py addsuperuser
   ```


## Configuring the LLM backend

- **Claude (Anthropic)**:
  - Set `LLM_CHOICE=CLAUDE` and define `ANTHROPIC_API_KEY` in `.env`.
- **OpenAI (hosted)**:
  - Set `LLM_CHOICE=OPENAI` and define `OPENAI_API_KEY`.
- **Gemini (Google)**:
  - Set `LLM_CHOICE=GEMINI` and define `GEMINI_API_KEY`.
- **Local / offline LLMs via Ollama**:
  - Install and run Ollama on the host.
  - Ensure the Docker Compose setup includes an `ollama` service (provided in this repo) reachable at `http://ollama:11434` from the web container.
  - Pull the desired models in Ollama, for example:
    ```bash
    ollama pull ebdm/gemma3-enhanced:12b
    ollama pull llama3.2
    ollama pull gpt-oss:120b
    ```
  - Set one of:
    - `LLM_CHOICE=GEMMA3`
    - `LLM_CHOICE=LLAMA3.2`
    - `LLM_CHOICE=GPT-OSS`
  - No external API keys are needed for these choices; the backend talks to Ollama via its OpenAI-compatible HTTP API.

### Embeddings and reranking

- **Document embeddings**:
  - AquiLLM now uses a local [SentenceTransformers](https://www.sbert.net/) model (`BAAI/bge-large-en-v1.5`) defined in `aquillm/utils.py`.
  - The model will be downloaded automatically on first use; ensure the machine has enough disk space and RAM.
- **Reranking**:
  - Candidate chunks from pgvector and trigram search are reranked using a local CrossEncoder (`BAAI/bge-reranker-base`) in `TextChunk.rerank`.
  - No Cohere rerank API calls are required anymore; everything runs locally.


## Using AquiLLM

### Uploading Documents

1. **Add a Collection First**
   - Click on "Collections" in the navigation menu
   - Click the "New Collection" button
   - Enter a name for your collection

2. **Upload Documents**
   - Go to the collection you created
   - Choose the document type you want to upload using the buttons, the buttons are as follows, in order from left to right:
     - PDF: Upload PDF files
     - ArXiv Paper: Enter an arXiv ID to import 
     - VTT File: Upload VTT transcript files
     - Webpage: Enter the URL of a site 
     - Handwritten Notes: Upload images of handwritten notes, select the Convert to LaTeX box if they contain formulas
     - All documents will appear in your collection, if they don't show up automatically, refresh the page
     
3. **View Your Documents**
   - If you leave your collection page, do the following
   - Select Collections button from the sidebar and choose which collection you want to view
   - Click on any document to view its contents

### Chatting With Your Documents

1. **Start a New Conversation**
   - From the sidebar, click "New Conversation"
   - Select which collections to include in your search context

2. **Using the Chat**
   - Type your questions about the documents in natural language
   - The AI will search your documents and provide answers with references
   - You can follow up with additional questions
   - The AI may quote specific parts of your documents as references

3. **Managing Conversations**
   - All conversations are saved automatically
   - Access past conversations from the "Your Conversations" menu in the sidebar
   - Each conversation maintains its collection context

## Contributing

We welcome contributions! AquiLLM is an open-source project, and we appreciate help from the community.

*   **Reporting Bugs**: Please open an issue on GitHub detailing the problem, expected behavior, and steps to reproduce.
*   **Feature Requests**: Open an issue describing the feature and its potential benefits.
*   **Pull Requests**: Send a pull request!


## Changelog

### 2026-03-06 – Local LLMs, local embeddings, agentic tools

- **Local embeddings**: Switched RAG embeddings from Cohere to a local SentenceTransformers model (`BAAI/bge-large-en-v1.5`) and added a local CrossEncoder reranker (`BAAI/bge-reranker-base`) for higher-quality ranking without external API calls.
- **LLM backends**: Extended `LLM_CHOICE` to support Ollama-backed models (`GEMMA3`, `LLAMA3.2`, `GPT-OSS`) alongside existing Claude, OpenAI, and Gemini backends.
- **Agentic tool loop**: Updated the chat backend to run an internal agent loop that automatically executes tools (vector search, whole_document, astrophotography helpers, etc.) and only returns final conversational text to the user.
- **Tool-call robustness**: Hardened tool-call parsing so local models that emit raw JSON or markdown-wrapped tool calls are correctly interpreted and executed, instead of leaking raw JSON into the chat UI.
