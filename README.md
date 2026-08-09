# PDF RAG Assistant

A modern RAG (Retrieval-Augmented Generation) web app that lets you upload PDF documents and ask questions about their content. Answers are generated **only** from the uploaded documents using vector search + a large language model.

![Stack](https://img.shields.io/badge/Python-3.14-blue) ![Framework](https://img.shields.io/badge/FastAPI-0.141-green) ![VectorDB](https://img.shields.io/badge/ChromaDB-1.5-orange) ![UI](https://img.shields.io/badge/UI-Vanilla%20JS-9cf)

## Features

- **Upload PDFs** — drag & drop or click to upload one or many PDFs (duplicate detection via SHA-256)
- **Smart chat** — ask questions; answers are grounded only in your uploaded documents
- **Source citations** — every answer shows which PDF and page the information came from
- **Modern neon UI** — dark futuristic theme with glassmorphism, animations, light/dark toggle
- **Local embeddings** — `all-MiniLM-L6-v2` runs on your machine; no external embedding API needed
- **Persistent vector store** — Chroma DB keeps your indexed documents across restarts

## Tech Stack

| Layer      | Technology                                     |
| ---------- | ---------------------------------------------- |
| Backend    | Python 3.14, FastAPI, Uvicorn                  |
| LLM        | Mistral AI (`mistral-small-2603`)              |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`       |
| Vector DB  | ChromaDB                                       |
| Documents  | PyPDFLoader + RecursiveCharacterTextSplitter   |
| Frontend   | Vanilla HTML / CSS / JS (no build step)        |

## Getting Started

### Prerequisites

- Python 3.11+
- A [Mistral AI API key](https://console.mistral.ai)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/pdf-rag-assistant.git
cd pdf-rag-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Mistral API key
# Create a .env file in the project root:
echo "MISTRAL_API_KEY=your_key_here" > .env
```

### Run the app

```bash
uvicorn app:app --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a PDF, and start asking questions.

## Usage

1. **Upload** — drag a PDF onto the upload card or click to browse. Multiple files are supported.
2. **Ask** — once a document is indexed, type a question in the chat box.
3. **Clear** — use the "Clear" button in the header to reset the conversation.

The assistant will answer strictly from your documents. If the answer is not in the context, it replies: *"I could not find the answer in the document."*

## Project Structure

```
.
├── app.py                # FastAPI application (main entry point)
├── requirements.txt      # Pinned Python dependencies
├── render.yaml           # Render Blueprint (deployment config)
├── .gitignore
├── static/
│   ├── index.html        # Frontend UI
│   ├── style.css         # Styling & animations
│   └── script.js         # Frontend logic
├── create_db.py          # Dev helper: build a Chroma DB from a PDF
├── main.py               # Dev helper: CLI-based Q&A loop
└── chroma-db/            # Vector store (created at runtime)
```

## API Endpoints

| Method | Path          | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| GET    | `/`           | Serves the web UI                            |
| POST   | `/api/upload` | Uploads a PDF (`multipart/form-data`, field `file`) |
| POST   | `/api/chat`   | Asks a question (`{"question": "..."}`)      |

### Example: chat request

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is an RNN?"}'
```

## Deployment on Render

This repo includes a [Render Blueprint](render.yaml) for one-click deployment.

1. Push this repo to GitHub
2. On [render.com](https://render.com): **New → Blueprint →** connect your repo
3. Set the `MISTRAL_API_KEY` secret when prompted
4. Click **Deploy** — your app is live at `https://pdf-rag-assistant.onrender.com`

> Note: The free tier uses an ephemeral filesystem. Uploaded documents and the vector store reset on each redeploy, and the service sleeps after ~15 minutes of inactivity (first request after wake takes ~30–60s).

## Environment Variables

| Variable          | Required | Description                         |
| ----------------- | -------- | ----------------------------------- |
| `MISTRAL_API_KEY` | Yes      | Mistral AI API key for the LLM      |

## Roadmap

- [ ] Persistent disk for uploaded documents on Render
- [ ] Conversation history across sessions
- [ ] Support for additional file types (TXT, DOCX, Markdown)
- [ ] Streaming responses

## License

MIT
