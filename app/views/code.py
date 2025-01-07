from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tiktoken
from data_reader import DataReader
from embedding import EmbeddingsGenerator
from vector_store import VectorStore

import os

def main():
    # Key for OpenAI API

    # Step 1: Initialize components
    pdf_path = "Data_LLM.pdf"  
    reader = DataReader(pdf_path)
    embeddings_generator = EmbeddingsGenerator()

    # Step 2: Process the data
    pdf_text = reader.read_pdf()
    docs = reader.split_text_into_chunks(pdf_text)

    # Step 3: Generate embeddings and vector store
    vector_store = VectorStore(embeddings_generator.embeddings)
    vector_store.create_vector_store(docs)

    # Step 4: Create retriever and QA chain
    retriever = vector_store.get_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("Simple Chatbot (type 'exit' to quit)")

    # Step 5: Chatbot interaction
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        response = qa_chain.run(query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
