import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# PDF files and their corresponding store names
PDF_FILES = {
    "data_engineer": "/path/to/Data_Engineer.pdf",
    "software_engineer": "/path/to/Software_Engineer.pdf",
    "platform_engineer": "/path/to/Platform_Engineer.pdf",
}

# Directory to store Chroma vector stores
CHROMA_DIR = "./chroma"


### Utility Functions ###
def validate_file_path(file_path):
    """Check if the file path exists."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False
    return True


def read_pdf(pdf_path):
    """
    Read and extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
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
    """
    Split text into manageable chunks.

    Args:
        text (str): Input text to split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


def create_vector_store(docs, store_name):
    """
    Create a Chroma vector store.

    Args:
        docs (list): List of text chunks.
        store_name (str): Name of the vector store.

    Returns:
        None
    """
    try:
        persist_dir = os.path.join(CHROMA_DIR, store_name)
        embeddings = OpenAIEmbeddings()

        # Create Chroma vector store
        vector_store = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            collection_name=store_name,
            persist_directory=persist_dir,
        )
        vector_store.persist()  # Save the vector store to disk
        print(f"Vector store for '{store_name}' created and persisted at {persist_dir}")
    except Exception as e:
        print(f"Error creating vector store for {store_name}: {e}")


def load_vector_store(store_name):
    """
    Load an existing Chroma vector store.

    Args:
        store_name (str): Name of the vector store.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    try:
        persist_dir = os.path.join(CHROMA_DIR, store_name)
        embeddings = OpenAIEmbeddings()
        print(f"Loading vector store for '{store_name}' from {persist_dir}...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    except Exception as e:
        print(f"Error loading vector store for {store_name}: {e}")
        return None


def get_retriever_and_qa_chain(store_name):
    """
    Initialize a retriever and QA chain for the given store.

    Args:
        store_name (str): Name of the vector store.

    Returns:
        RetrievalQA: A QA chain ready to process queries.
    """
    vector_store = load_vector_store(store_name)
    if vector_store is None:
        raise ValueError(f"Failed to load vector store for {store_name}")

    retriever = vector_store.as_retriever()

    # Initialize LLM and QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain


### PDF Processing ###
def process_all_pdfs():
    """
    Process all PDFs and store them in vector stores.

    Returns:
        None
    """
    print("\nStarting PDF processing and storing in ChromaDB...")
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

        # Create a vector store for the bucket
        create_vector_store(text_chunks, store_name)

    print("\nAll PDFs have been processed and stored successfully in ChromaDB.")


### Interactive Chat ###
def interactive_chat(store_name):
    """
    Start an interactive chatbot for a given vector store.

    Args:
        store_name (str): Name of the vector store.

    Returns:
        None
    """
    try:
        print(f"\nStarting Chatbot for '{store_name}' bucket. (Type 'exit' to quit)...")
        qa_chain = get_retriever_and_qa_chain(store_name)

        # Chatbot loop
        while True:
            query = input("\nAsk a question: ")
            if query.lower() == "exit":
                print("Returning to main menu...")
                break

            try:
                # Run the query and get a response
                response = qa_chain.run(query)

                # Validate if the response is relevant or empty
                if not response.strip():
                    print(f"No relevant information found in the '{store_name}' bucket.")
                else:
                    print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {str(e)}")
    except ValueError as e:
        print(e)


### Main Function ###
def main():
    """
    Main function to manage the chatbot application workflow.
    """
    print("=" * 80)
    print("Welcome to the Domain-Specific Document Query System")
    print("This system processes and stores documents, enabling efficient querying.")
    print("=" * 80)

    # Step 1: Process and store all PDFs into ChromaDB
    process_all_pdfs()

    # Step 2: User interaction for querying
    while True:
        print("\nAvailable buckets for querying:")
        for bucket in PDF_FILES.keys():
            print(f"- {bucket}")

        print("\nType 'exit' to quit the application at any time.")
        selected_bucket = input("\nSelect a bucket to query (data_engineer/software_engineer/platform_engineer): ").strip().lower()

        # Handle exit scenario
        if selected_bucket == "exit":
            print("\nThank you for using the Document Query System. Goodbye!")
            break

        # Validate the selected bucket
        if selected_bucket not in PDF_FILES:
            print("\nInvalid bucket name. Please choose a valid option from the list.")
            continue

        # Query the selected bucket
        interactive_chat(selected_bucket)


### Run the Script ###
if __name__ == "__main__":
    main()
