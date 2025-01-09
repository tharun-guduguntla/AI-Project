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

# Path to PDF folder
PDF_FOLDER = "./pdfs"
PDF_FILES = {
    "data_engineer": os.path.join(PDF_FOLDER, "Data_Engineer.pdf"),
    "software_engineer": os.path.join(PDF_FOLDER, "Software_Engineer.pdf"),
    "platform_engineer": os.path.join(PDF_FOLDER, "Platform_Engineer.pdf"),
}

# Initialize ChromaDB
chroma_client = Client(Settings(persist_directory="./chroma_store"))

### Utility Functions ###
def validate_file_path(file_path):
    """Check if the file path exists."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    return True

def read_pdf(pdf_path):
    """Read and extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        return pdf_text
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=300, chunk_overlap=100):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def generate_embeddings(chunks):
    """Generate embeddings using Vertex AI."""
    embeddings = []
    for chunk in chunks:
        try:
            response = GENAI_MODEL.predict(chunk)
            embedding = response.text_embedding  # Adjust as per the SDK documentation
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
    return embeddings

def create_vector_store_in_chroma(store_name, chunks, embeddings):
    """
    Store embeddings into ChromaDB.

    Args:
        store_name (str): Name of the bucket.
        chunks (list): List of text chunks.
        embeddings (list): Embeddings for the chunks.

    Returns:
        None
    """
    try:
        collection = chroma_client.get_or_create_collection(name=store_name)
        for chunk, embedding in zip(chunks, embeddings):
            collection.add(documents=[chunk], embeddings=[embedding])
        print(f"Data stored successfully in ChromaDB for {store_name}.")
    except Exception as e:
        print(f"Error storing data in ChromaDB: {e}")

def retrieve_from_chroma(store_name, query_embedding, top_k=5):
    """
    Retrieve relevant documents from ChromaDB using embeddings.

    Args:
        store_name (str): Name of the bucket.
        query_embedding (list): Query embedding.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Relevant text chunks.
    """
    try:
        collection = chroma_client.get_collection(name=store_name)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results.get("documents", [])
    except Exception as e:
        print(f"Error retrieving data from ChromaDB: {e}")
        return []

### Main Functions ###
def process_all_pdfs():
    """
    Process all PDFs and store them in ChromaDB.
    """
    print("\nProcessing all PDFs and storing data in ChromaDB...")

    for store_name, pdf_path in PDF_FILES.items():
        print(f"\nProcessing '{store_name}' bucket...")

        if not validate_file_path(pdf_path):
            print(f"Skipping '{store_name}' bucket due to missing file.")
            continue

        pdf_text = read_pdf(pdf_path)
        if not pdf_text:
            print(f"Skipping '{store_name}' bucket due to empty content in {pdf_path}.")
            continue

        text_chunks = split_text_into_chunks(pdf_text)
        embeddings = generate_embeddings(text_chunks)
        create_vector_store_in_chroma(store_name, text_chunks, embeddings)

    print("\nAll PDFs have been processed and stored successfully.")

def interactive_chat():
    """
    Allow the user to select a bucket and ask questions.
    """
    while True:
        print("\nAvailable buckets for querying:")
        for bucket in PDF_FILES.keys():
            print(f"- {bucket}")

        selected_bucket = input("\nSelect a bucket to query (data_engineer/software_engineer/platform_engineer) or 'exit' to quit: ").strip().lower()
        if selected_bucket == "exit":
            print("\nThank you for using the Agentic AI System. Goodbye!")
            break

        if selected_bucket not in PDF_FILES:
            print("\nInvalid bucket name. Please choose a valid option.")
            continue

        print(f"\nStarting Chatbot for '{selected_bucket}' bucket. (Type 'exit' to quit)...")

        while True:
            query = input("\nAsk a question: ")
            if query.lower() == "exit":
                print("Returning to bucket selection...")
                break

            query_embedding = GENAI_MODEL.predict(query).text_embedding  # Adjust attribute if needed
            results = retrieve_from_chroma(selected_bucket, query_embedding)

            if not results:
                print(f"No relevant information found in the '{selected_bucket}' bucket.")
            else:
                print(f"Top Results:\n{results}")

def main():
    print("=" * 80)
    print("Welcome to the Agentic AI-Powered Document Query System with ChromaDB")
    print("This system processes and stores documents, enabling efficient querying by job family.")
    print("=" * 80)

    process_all_pdfs()
    interactive_chat()

if __name__ == "__main__":
    main()
