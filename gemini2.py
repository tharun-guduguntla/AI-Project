import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Set up Google API key
GENAI_API_KEY = "your_google_api_key"
os.environ["GENAI_API_KEY"] = GENAI_API_KEY

genai.configure(api_key=GENAI_API_KEY)

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
    """Generate embeddings using Google Generative AI."""
    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = embeddings_model.embed_documents(chunks)
        print(f"Generated {len(embeddings)} embeddings successfully.")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []


### Main Functions ###
def process_all_pdfs():
    """
    Process all PDFs and generate embeddings.
    """
    print("\nProcessing all PDFs...")

    PDF_FILES = {
        "data_engineer": "Data_Engineer.pdf",
        "software_engineer": "Software_Engineer.pdf",
        "platform_engineer": "Platform_Engineer.pdf",
    }

    for store_name, pdf_path in PDF_FILES.items():
        print(f"\nProcessing '{store_name}' bucket...")

        # Validate the file path
        if not validate_file_path(pdf_path):
            print(f"Skipping '{store_name}' bucket due to missing file.")
            continue

        # Read the PDF content
        pdf_text = read_pdf(pdf_path)

        if not pdf_text:
            print(f"Skipping '{store_name}' bucket due to empty content in {pdf_path}.")
            continue

        # Split the PDF text into chunks
        text_chunks = split_text_into_chunks(pdf_text)

        # Generate embeddings for the chunks
        embeddings = generate_embeddings(text_chunks)

        # Uncomment the following if you want to include AlloyDB operations
        # create_vector_store_in_alloydb(store_name, text_chunks, embeddings)

    print("\nAll PDFs have been processed.")


def interactive_chat(store_name):
    """
    Start an interactive chatbot for a given bucket.
    """
    print(f"\nStarting Chatbot for '{store_name}' bucket. (Type 'exit' to quit)...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            print("Returning to main menu...")
            break

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)

        # Uncomment the following if you want to retrieve data from AlloyDB
        # results = retrieve_from_alloydb(store_name, query_embedding)

        # Simulating result retrieval
        results = ["Sample response 1", "Sample response 2"]

        if not results:
            print(f"No relevant information found in the '{store_name}' bucket.")
        else:
            print(f"Top Results:\n{results}")


### Main Function ###
def main():
    print("=" * 80)
    print("Welcome to the Gemini-Powered Document Query System")
    print("This system processes and generates embeddings for documents.")
    print("=" * 80)

    # Step 1: Process PDFs
    process_all_pdfs()

    # Step 2: Simulated interaction
    while True:
        store_name = input("\nEnter a bucket name (data_engineer/software_engineer/platform_engineer) or 'exit' to quit: ").strip().lower()
        if store_name == "exit":
            print("\nThank you for using the Document Query System. Goodbye!")
            break
        interactive_chat(store_name)


if __name__ == "__main__":
    main()