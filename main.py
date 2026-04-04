import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from utils import split_text, open_pdf, store_embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI



app = FastAPI()
os.makedirs("data", exist_ok=True)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
embeddings = OpenAIEmbeddings()


CURRENT_DOCUMENT = None

@app.get("/")
def read_root():
    return {"message": "server is running"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename)
    file_location = f"data/{safe_filename}"
    global CURRENT_DOCUMENT
    CURRENT_DOCUMENT = file_location
    contents = file.file.read()

    if len(contents) > 5_000_000:
        return {"error":"File too Large"}


    with open(file_location, "wb") as f:
        f.write(contents)

    full_text = open_pdf(file_location)
    chunks = split_text(full_text, chunk_size=3000)
    store_embeddings(chunks, file_location)


    return {"info": f"file saved at {file_location}",
                "text_preview": full_text[:200],
                "total_length": len(full_text),
                "chunk_length": len(chunks),
                "chunks": chunks[:5]
                }

llm = ChatOpenAI(model="gpt-4o-mini")

@app.post("/ask")
def ask_question(question: str):
    if not question:
        return {"error": "question is required"}

    answer = generate_answer(question)
    return {"answer": answer}


def generate_answer(question):
    if not CURRENT_DOCUMENT:
        return "No document uploaded yet"

    db = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="db"
    )

    results = db.similarity_search(
        question,
        k=3,
        filter={"source": CURRENT_DOCUMENT}
    )

    context = ""
    for result in results:
        context += result.page_content + "\n\n"

    response = llm.invoke(
        f"Answer only from this context:\n{context}\n\nQuestion: {question}"
    )

    return response.content


@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    # Handle both normal and edited messages
    message = data.get("message") or data.get("edited_message", {})
    chat_id = message.get("chat", {}).get("id")

    # =========================
    # 📄 HANDLE DOCUMENT (PDF)
    # =========================
    document = message.get("document")

    if document:
        file_name = document.get("file_name", "")
        file_id = document.get("file_id")

        # Only allow PDF
        if not file_name.lower().endswith(".pdf"):
            requests.post(
                f"{TELEGRAM_API_URL}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": "❌ Please upload a PDF file only."
                }
            )
            return {"status": "invalid file"}

        # Step 1: Get file path
        file_info = requests.get(
            f"{TELEGRAM_API_URL}/getFile?file_id={file_id}"
        ).json()

        file_path = file_info["result"]["file_path"]

        # Step 2: Download file
        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
        file_data = requests.get(file_url).content

        file_location = f"data/{file_name}"

        with open(file_location, "wb") as f:
            f.write(file_data)

        # Step 3: Process PDF
        full_text = open_pdf(file_location)
        chunks = split_text(full_text)
        store_embeddings(chunks, file_location)

        # Step 4: Set current document
        global CURRENT_DOCUMENT
        CURRENT_DOCUMENT = file_location

        # Step 5: Confirm upload
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "✅ PDF uploaded successfully. Ask your question."
            }
        )

        return {"status": "document processed"}

    # =========================
    # 💬 HANDLE TEXT MESSAGE
    # =========================
    text = message.get("text")

    if not text:
        return {"status": "no text"}

    answer = generate_answer(text)

    requests.post(
        f"{TELEGRAM_API_URL}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": answer[:3000]  # Telegram limit safety
        }
    )

    return {"status": "ok"}




















