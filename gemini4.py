import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from google.cloud import alloydb_v1beta
from vertexai.preview.language_models import GenerativeModel

# Initialize Vertex AI and Gemini Model
import vertexai
vertexai.init(project="playpen-33fcd2", location="europe-central12")
GENAI_MODEL = GenerativeModel("gemini-1.5-fash-002")

# AlloyDB Credentials
ALLOYDB_HOST = "your-alloydb-instance-ip"
ALLOYDB_PORT = "5432"
ALLOYDB_USER = "your-username"
ALLOYDB_PASSWORD = "your-password"
ALLOYDB_DATABASE = "your-database"

# Path to PDF folder
PDF_FOLDER = "./pdfs"
PDF_FILES = {
    "data_engineer": os.path.join(PDF_FOLDER, "Data_Engineer.pdf"),
    "software_engineer": os.path.join(PDF_FOLDER, "Software_Engineer.pdf"),
    "platform_engineer": os.path.join(PDF_FOLDER, "Platform_Engineer.pdf"),
}

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

def create_vector_store_in_alloydb(store_name, chunks, embeddings):
    """
    Store embeddings into AlloyDB.
   
    Args:
        store_name (str): Name of the bucket.
        chunks (list): List of text chunks.
        embeddings (list): Embeddings for the chunks.

    Returns:
        None
    """
    try:
        import psycopg2
        connection = psycopg2.connect(
            host=ALLOYDB_HOST,
            port=ALLOYDB_PORT,
            user=ALLOYDB_USER,
            password=ALLOYDB_PASSWORD,
            dbname=ALLOYDB_DATABASE,
        )
        cursor = connection.cursor()

        # Create table for the vector store
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {store_name} (
            id SERIAL PRIMARY KEY,
            text_chunk TEXT,
            embedding VECTOR(768)  -- Assuming 768 dimensions for embeddings
        );
        """)
        connection.commit()

        # Insert chunks and embeddings into the table
        for chunk, embedding in zip(chunks, embeddings):
            cursor.execute(
                f"INSERT INTO {store_name} (text_chunk, embedding) VALUES (%s, %s)",
                (chunk, embedding.tolist()),
            )
        connection.commit()

        print(f"Data stored successfully in AlloyDB for {store_name}.")
    except Exception as e:
        print(f"Error storing data in AlloyDB: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def retrieve_from_alloydb(store_name, query_embedding, top_k=5):
    """
    Retrieve relevant documents from AlloyDB using embeddings.
   
    Args:
        store_name (str): Name of the bucket.
        query_embedding (list): Query embedding.
        top_k (int): Number of results to retrieve.

    Returns:
        list: Relevant text chunks.
    """
    try:
        import psycopg2
        connection = psycopg2.connect(
            host=ALLOYDB_HOST,
            port=ALLOYDB_PORT,
            user=ALLOYDB_USER,
            password=ALLOYDB_PASSWORD,
            dbname=ALLOYDB_DATABASE,
        )
        cursor = connection.cursor()

        # Use a similarity function (cosine similarity) for retrieval
        cursor.execute(f"""
        SELECT text_chunk
        FROM {store_name}
        ORDER BY embedding <-> %s
        LIMIT {top_k};
        """, (query_embedding,))
        results = cursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        print(f"Error retrieving data from AlloyDB: {e}")
        return []
    finally:
        if connection:
            cursor.close()
            connection.close()

def generate_embeddings(chunks):
    """Generate embeddings using Vertex AI."""
    embeddings = []
    for chunk in chunks:
        response = GENAI_MODEL.get_embeddings([chunk])
        embeddings.append(response.embeddings[0])
    return embeddings

### Main Functions ###
def process_all_pdfs():
    """
    Process all PDFs and store them in AlloyDB.
    """
    print("\nProcessing all PDFs and storing data in AlloyDB...")

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

        # Store the data in AlloyDB
        create_vector_store_in_alloydb(store_name, text_chunks, embeddings)

    print("\nAll PDFs have been processed and stored successfully.")

def interactive_chat(store_name):
    """
    Start an interactive chatbot for a given bucket.
    """
    print(f"\nStarting Chatbot for '{store_name}' bucket. (Type 'exit' to quit)...")

    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            print("Returning to main menu...")
            break

        # Generate query embedding
        query_embedding = GENAI_MODEL.get_embeddings([query]).embeddings[0]

        # Retrieve relevant text chunks from AlloyDB
        results = retrieve_from_alloydb(store_name, query_embedding)

        if not results:
            print(f"No relevant information found in the '{store_name}' bucket.")
        else:
            print(f"Top Results:\n{results}")

### Main Function ###
def main():
    print("=" * 80)
    print("Welcome to the Gemini-Powered Document Query System with AlloyDB")
    print("This system processes and stores documents, enabling efficient querying.")
    print("=" * 80)

    # Step 1: Process PDFs and store data in AlloyDB
    process_all_pdfs()

    # Step 2: User interaction for querying
    while True:
        print("\nAvailable buckets for querying:")
        for bucket in PDF_FILES.keys():
            print(f"- {bucket}")

        selected_bucket = input("\nSelect a bucket to query (data_engineer/software_engineer/platform_engineer): ").strip().lower()
        if selected_bucket == "exit":
            print("\nThank you for using the Document Query System. Goodbye!")
            break

        if selected_bucket not in PDF_FILES:
            print("\nInvalid bucket name. Please choose a valid option.")
            continue

        # Start chatbot for the selected bucket
        interactive_chat(selected_bucket)

if __name__ == "__main__":
    main()