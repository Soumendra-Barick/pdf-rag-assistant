import hashlib
import json
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CHROMA_DIR = BASE_DIR / "chroma-db"
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_FILE = BASE_DIR / "indexed_files.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVE = 4

app = FastAPI(title="PDF RAG Assistant")

UPLOAD_DIR.mkdir(exist_ok=True)


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code: int = 200) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": K_RETRIEVE,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

llm = ChatMistralAI(model="mistral-small-2603")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say: "I could not find the answer in the document."
            """,
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """,
        ),
    ]
)

index_lock = threading.Lock()
jobs: dict = {}


def load_index() -> dict:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_index(index: dict) -> None:
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_name = Path(file.filename).name
    temp_path = None

    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False, dir=str(UPLOAD_DIR)
    ) as tmp:
        temp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)

    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "processing",
        "message": "Indexing PDF...",
        "filename": safe_name,
    }
    background_tasks.add_task(process_pdf, job_id, temp_path, safe_name)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/upload/{job_id}")
def upload_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def process_pdf(job_id: str, temp_path: Path, safe_name: str) -> None:
    try:
        file_hash = file_sha256(temp_path)

        with index_lock:
            index = load_index()
            if file_hash in index:
                temp_path.unlink(missing_ok=True)
                jobs[job_id] = {
                    "status": "duplicate",
                    "message": f"'{safe_name}' is already uploaded.",
                    "original": index[file_hash],
                }
                return

            jobs[job_id] = {
                "status": "processing",
                "message": "Reading pages...",
                "filename": safe_name,
            }

            loader = PyPDFLoader(str(temp_path))
            docs = loader.load()

            if not docs:
                temp_path.unlink(missing_ok=True)
                jobs[job_id] = {
                    "status": "error",
                    "message": "The PDF appears to be empty.",
                }
                return

            jobs[job_id] = {
                "status": "processing",
                "message": "Splitting into chunks...",
                "filename": safe_name,
            }

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            chunks = splitter.split_documents(docs)

            jobs[job_id] = {
                "status": "processing",
                "message": f"Embedding {len(chunks)} chunks (this can take a minute for large PDFs)...",
                "filename": safe_name,
            }

            vectorstore.add_documents(documents=chunks)
            index[file_hash] = safe_name
            save_index(index)

        jobs[job_id] = {
            "status": "done",
            "message": f"Uploaded '{safe_name}' successfully.",
            "chunks": len(chunks),
            "pages": len(docs),
        }
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        jobs[job_id] = {
            "status": "error",
            "message": f"Could not read PDF: {exc}",
        }


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = retriever.invoke(question)

    if not docs:
        return {"answer": "I could not find the answer in the document."}

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)

    return {
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
            }
            for doc in docs
        ],
    }
