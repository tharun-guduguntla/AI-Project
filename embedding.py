from langchain_community.embeddings import OpenAIEmbeddings

class EmbeddingsGenerator:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def generate_embeddings(self, docs):
        return self.embeddings.embed_documents(docs)
