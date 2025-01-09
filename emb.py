import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

# Initialize Vertex AI
aiplatform.init(project="playpen-33fcd2", location="europe-central12")
MODEL = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Path to PDF folder
PDF_FOLDER = "./pdfs"
PDF_FILES = {
    "data_engineer": os.path.join(PDF_FOLDER, "Data_Engineer.pdf"),
    "software_engineer": os.path.join(PDF_FOLDER, "Software_Engineer.pdf"),
    "platform_engineer": os.path.join(PDF_FOLDER, "Platform_Engineer.pdf"),
}

# Temporary in-memory store for embeddings
EMBEDDINGS_STORE = {}

# Utility Functions
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
            response = MODEL.get_embeddings([chunk])
            embeddings.append(response.embeddings[0])
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
    return embeddings

def process_all_pdfs():
    """
    Process all PDFs and store their embeddings in the in-memory store.
    """
    print("\nProcessing all PDFs and generating embeddings...")

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

        # Store in in-memory dictionary
        EMBEDDINGS_STORE[store_name] = {
            "chunks": text_chunks,
            "embeddings": embeddings,
        }

    print("\nAll PDFs have been processed and embeddings generated successfully.")

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

        if selected_bucket not in EMBEDDINGS_STORE:
            print("\nInvalid bucket name or embeddings not found. Please choose a valid option.")
            continue

        print(f"\nStarting Chatbot for '{selected_bucket}' bucket. (Type 'exit' to quit)...")

        while True:
            query = input("\nAsk a question: ")
            if query.lower() == "exit":
                print("Returning to bucket selection...")
                break

            try:
                # Generate query embedding
                query_embedding = MODEL.get_embeddings([query]).embeddings[0]

                # Retrieve relevant chunks
                store = EMBEDDINGS_STORE[selected_bucket]
                chunks = store["chunks"]
                embeddings = store["embeddings"]

                # Calculate similarity (cosine similarity placeholder)
                similarities = [
                    sum(qe * ce for qe, ce in zip(query_embedding, chunk_emb))
                    for chunk_emb in embeddings
                ]

                # Get top results
                top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
                top_results = [chunks[i] for i in top_indices]

                print("\nTop Results:")
                for result in top_results:
                    print(f"- {result}")
            except Exception as e:
                print(f"Error during query processing: {e}")

def main():
    print("=" * 80)
    print("Welcome to the Agentic AI-Powered Document Query System")
    print("This system processes and stores documents, enabling efficient querying by job family.")
    print("=" * 80)

    process_all_pdfs()
    interactive_chat()

if __name__ == "__main__":
    main()

