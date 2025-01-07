from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from app.utils.data_reader import DataReader
from app.models.embedding import EmbeddingsGenerator
from app.controllers.vector_store import VectorStore
import tiktoken
import os,yaml
from app.controllers import CONFIG_FILE_PATH,set_config_location
from langchain_community.embeddings import OpenAIEmbeddings

config = set_config_location()
print("config values here is",config)



class openAIAPIKEY:
    def __init__(self):
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

class ChatbotService:
    def __init__(self, pdf_path):
        self.reader = DataReader(pdf_path)
        self.embeddings_generator = openAIAPIKEY()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key= openai_api_key )
        self.qa_chain = None
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.initialize_service()

    def initialize_service(self):
        pdf_text = self.reader.read_pdf()
        docs = self.reader.split_text_into_chunks(pdf_text)
        vector_store = VectorStore(self.embeddings_generator.embeddings)
        vector_store.create_vector_store(docs)
        retriever = vector_store.get_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    def process_question(self, question):
        if not self.qa_chain:
            raise ValueError("QA Chain not initialized.")
        return self.qa_chain.run(question)
