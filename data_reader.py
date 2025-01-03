from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

class DataReader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def read_pdf(self):
        reader = PdfReader(self.pdf_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text()
        return pdf_text

    def split_text_into_chunks(self, text, chunk_size=300, chunk_overlap=100):
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
