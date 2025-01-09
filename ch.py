import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings
from vertexai.preview.language_models import TextGenerationModel

# Initialize Vertex AI
import vertexai
vertexai.init(project="playpen-33fcd2", location="europe-central12")
GENAI_MODEL = TextGenerationModel.from_pretrained("text-bison@001")

# ChromaDB Configuration
CHROMADB_DIR = "./chroma_db"
client = Client(Settings(persist_directory=CHROMADB_DIR))

PDF_FOLDER = "./pdfs"
PDF_FILES = {
    "data_engineer": os.path.join(PDF_FOLDER, "Data_Engineer.pdf"),
    "software_engineer": os.path.join(PDF_FOLDER, "Software_Engineer.pdf"),
    "platform_engineer": os.path.join(PDF_FOLDER, "Platform_Engineer.pdf"),
}

def process_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = GENAI_MODEL.predict(chunk)
        embeddings.append(response.text_embedding)
    return embeddings

def store_embeddings_in_chroma(bucket_name, chunks, embeddings):
    collection = client.create_collection(bucket_name)
    for i, chunk in enumerate(chunks):
        collection.add(f"doc-{i}", chunk, embeddings[i])

def interactive_chat(bucket_name):
    collection = client.get_collection(bucket_name)
    print(f"\nQuerying bucket: {bucket_name} (Type 'exit' to quit)")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        response = GENAI_MODEL.predict(query)
        embedding = response.text_embedding
        results = collection.query(embedding)
        print(f"Results: {results}")

def main():
    for bucket, file_path in PDF_FILES.items():
        if os.path.exists(file_path):
            text = process_pdf(file_path)
            chunks = CharacterTextSplitter().split_text(text)
            embeddings = generate_embeddings(chunks)
            store_embeddings_in_chroma(bucket, chunks, embeddings)

    while True:
        print("\nBuckets Available:", list(PDF_FILES.keys()))
        bucket = input("\nChoose a bucket to query: ")
        if bucket in PDF_FILES:
            interactive_chat(bucket)
        elif bucket.lower() == "exit":
            break
        else:
            print("Invalid bucket name.")

if __name__ == "__main__":
    main()
