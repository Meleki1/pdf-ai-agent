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


@app.get("/")
def read_root():
    return {"message": "server is running"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    safe_filename = os.path.basename(file.filename)
    file_location = f"data/{safe_filename}"

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
        return {"error": "question is requried"}
    db = Chroma(
        collection_name="documents", 
        embedding_function=embeddings, 
        persist_directory="db"
        )
    
    results = db.similarity_search(question, k=3)
    context = ""

    for result in results:
        context += result.page_content + "\n\n"
    
    response = llm.invoke(f"Answer the question based on this context:\n{context}\n\nQuestion: {question}")
    return {"answer": response.content }


@app.get("/test")
def test():
    return {"message": "test works"}

@app.post("/webhook")
async def telegram_webhook(req: Request):
    data = await req.json()

    # Get message text
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text")

    if not text:
        return {"status": "no text"}

    # 👉 Call your existing RAG logic
    db = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="db"
    )

    results = db.similarity_search(text, k=3)

    context = ""
    for result in results:
        context += result.page_content + "\n\n"

    prompt = f"""
    Answer based only on this context:
    {context}

    Question: {text}
    """

    response = llm.invoke(prompt)
    answer = response.content

    # 👉 Send reply back to Telegram
    requests.post(
        f"{TELEGRAM_API_URL}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": answer
        }
    )

    return {"status": "ok"}





















