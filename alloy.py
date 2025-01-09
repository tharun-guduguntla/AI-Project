import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

# Initialize Vertex AI
aiplatform.init(project="playpen-33fcd2", location="europe-central12")
EMBEDDING_MODEL = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Path to PDF folder
PDF_FOLDER = "./pdfs"
PDF_FILES = {
    "data_engineer": os.path.join(PDF_FOLDER, "Data_Engineer.pdf"),
    "software_engineer": os.path.join(PDF_FOLDER, "Software_Engineer.pdf"),
    "platform_engineer": os.path.join(PDF_FOLDER, "Platform_Engineer.pdf"),
}

# In-memory store for embeddings and text chunks
VECTOR_STORE = {}

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
    """Generate embeddings using Vertex AI's TextEmbeddingModel."""
    embeddings = []
    for chunk in chunks:
        try:
            response = EMBEDDING_MODEL.get_embeddings([chunk])
            embeddings.append(response[0].values)  # Ensure proper extraction
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
    return embeddings

### Main Functions ###
def process_all_pdfs():
    """
    Process all PDFs and store embeddings and text in an in-memory store.
    """
    print("\nProcessing all PDFs and storing data...")

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
        VECTOR_STORE[store_name] = {"chunks": text_chunks, "embeddings": embeddings}

    print("\nAll PDFs have been processed and stored successfully.")

def retrieve_relevant_chunks(store_name, query, top_k=5):
    """Retrieve relevant text chunks based on query embedding similarity."""
    if store_name not in VECTOR_STORE:
        print(f"Store '{store_name}' not found.")
        return []

    try:
        query_embedding = EMBEDDING_MODEL.get_embeddings([query])[0].values  # Proper extraction
        store_data = VECTOR_STORE[store_name]
        chunks = store_data["chunks"]
        embeddings = store_data["embeddings"]

        # Compute similarity (dot product or cosine similarity)
        similarities = [
            (i, sum(e * q for e, q in zip(embedding, query_embedding)))
            for i, embedding in enumerate(embeddings)
        ]
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in similarities[:top_k]]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        print(f"Error during query processing: {e}")
        return []

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

        if selected_bucket not in VECTOR_STORE:
            print("\nInvalid bucket name. Please choose a valid option.")
            continue

        print(f"\nStarting Chatbot for '{selected_bucket}' bucket. (Type 'exit' to quit)...")

        while True:
            query = input("\nAsk a question: ")
            if query.lower() == "exit":
                print("Returning to bucket selection...")
                break

            results = retrieve_relevant_chunks(selected_bucket, query)

            if not results:
                print(f"No relevant information found in the '{selected_bucket}' bucket.")
            else:
                print("Top Results:")
                for result in results:
                    print(f"- {result}")

def main():
    print("=" * 80)
    print("Welcome to the Agentic AI-Powered Document Query System")
    print("This system processes and stores documents, enabling efficient querying by job family.")
    print("=" * 80)

    process_all_pdfs()
    interactive_chat()

if __name__ == "__main__":
    main()////////////////////
    def main():
    print("=" * 80)
    print("Welcome to the Agentic AI-Powered Document Query System")
    print("This system processes and stores documents, enabling efficient querying by job family.")
    print("=" * 80)

    # Step 1: Process PDFs and store data
    process_all_pdfs()

    # Step 2: Print the stored data
    print("\nData stored in VECTOR_STORE:")
    for store_name, store_data in VECTOR_STORE.items():
        print(f"\nBucket: {store_name}")
        print(f"Number of chunks: {len(store_data['chunks'])}")
        print(f"Number of embeddings: {len(store_data['embeddings'])}")
        print("Sample Chunk:")
        if store_data["chunks"]:
            print(f"- {store_data['chunks'][0]}")
        print("Sample Embedding:")
        if store_data["embeddings"]:
            print(f"- {store_data['embeddings'][0][:5]}...")  # Print first 5 values of the embedding

    # Step 3: Interactive Chat
    interactive_chat()

