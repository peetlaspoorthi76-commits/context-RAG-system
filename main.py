from fastapi import FastAPI, File, UploadFile,HTTPException
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Generator
from fastembed import TextEmbedding
import uuid
import os

from dotenv import load_dotenv
from groq import Groq

# Load the secret variables from the .env file
load_dotenv()

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

myapp = FastAPI()

embedding_model = TextEmbedding(model_name = "BAAI/bge-small-en-v1.5")

qdrant_client = AsyncQdrantClient(url="http://localhost:6333")

# Initialize the Groq client using the hidden API key
api_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

COLLECTION_NAME = "pdf_knowledge_base"

@myapp.on_event("startup")
async def initialize_database():
    """Runs when the server starts up to ensure our storage room exists"""
    # Check if our target collection folder already exists in the database
    try:
        exists = await qdrant_client.collection_exists(collection_name=COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            "Qdrant is not running. Start Qdrant server first at http://localhost:6333"
        )

    if not exists:
        # Create a new storage collection configured to match our embedding output dimensions
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # BAAI/bge-small-en-v1.5 model exports exactly 384 dimensions
                distance=Distance.COSINE # Use Cosine calculations to judge text similarity
            )
        )


def stream_uploaded_chunks(file_stream, chunk_size:int = 1000, overlap : int = 200)->Generator[str,None,None]:
    reader = PdfReader(file_stream)

    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,chunk_overlap = overlap,length_function = len,
        separators = ["\n\n","\n"," ",""])

    chunks_iterator = splitter.split_text(full_text)

    for chunk in chunks_iterator:
        yield chunk

@myapp.post('/upload-files/')
async def upload_and_embed_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    chunks_generator = stream_uploaded_chunks(file.file)

    chunk_count = 0

    for chunk_text in chunks_generator:
        chunk_count += 1

        try:
            vector_generator = embedding_model.embed([chunk_text])

            vector_embedding = next(vector_generator).tolist()

            point_id = str(uuid.uuid4())

            # 4. Package your vector numbers, text content, and file tracking metadata
            # into a structural data point required by the database engine
            data_point = PointStruct(
                id=point_id,
                vector=vector_embedding,
                payload={
                    "text": chunk_text,
                    "source_file": file.filename,
                    "chunk_index": chunk_count
                }
            )

            # 5. Commit the data point directly to your database cluster instantly
            await qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[data_point]
            )

        except Exception as e:
            print(f"Failed to process chunk {chunk_count}: {str(e)}")
            continue

    return {
        "filename": file.filename,
        "total_chunks": chunk_count,
        "status": "Successful"
    }

@myapp.post('/queries/')
async def user_query(query: str):
    # 1. Embed the search query
    query_embeddings = list(embedding_model.query_embed(query))
    query_vector = query_embeddings[0].tolist()

    # 2. Search Qdrant for the most relevant PDF chunk
    results = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1,
        score_threshold=0.5
    )

    if not results.points:
        return {"status": "Failed", "message": "I couldn't find anything relevant in the PDF."}

    # 3. Extract the context text from the database
    matched_point = results.points[0]
    context_text = matched_point.payload.get("text")
    source_file = matched_point.payload.get("source_file")

    # 4. Build the RAG Prompt
    rag_prompt = f"""
    You are a helpful assistant. Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context_text}

    User Question:
    {query}
    """

    # 5. Send it to the cloud API (using Llama 3)
    chat_completion = api_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": rag_prompt,
            }
        ],
        model="llama-3.1-8b-instant", # This is Meta's fast, open-source model hosted by Groq
    )

    # 6. Return the final generated answer
    return {
        "status": "Successful",
        "answer": chat_completion.choices[0].message.content,
        "source_used": source_file
    }