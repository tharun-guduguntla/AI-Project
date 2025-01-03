from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vector_store = None

    def create_vector_store(self, docs):
        self.vector_store = Chroma.from_texts(docs, self.embeddings)
        return self.vector_store

    def get_retriever(self):
        if self.vector_store:
            return self.vector_store.as_retriever()
        else:
            raise ValueError("Vector store not initialized!")
