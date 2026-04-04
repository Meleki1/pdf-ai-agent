import os
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma




load_dotenv()
embeddings = OpenAIEmbeddings()
api_key = os.getenv("OPENAI_API_KEY")




def open_pdf(file_location):
    reader = PdfReader(file_location)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "/n"
    return full_text


def split_text(full_text, chunk_size=3000):
    chunk = []
    for i in range(0, len(full_text), chunk_size):
        chunk.append(full_text[i:i+chunk_size])
    return chunk




def store_embeddings(chunks, file_location):
    db = Chroma(
        collection_name="documents", 
        embedding_function=embeddings, 
        persist_directory="db"
        )

    print(f"Number of chunks: {len(chunks)}")

    db.add_texts(
        chunks, 
        metadatas=[{"source": file_location}] * len(chunks)
        )

    print("Embeddings stored successfully!")

