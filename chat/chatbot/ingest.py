import os
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from chatapp.models import Document

EMB = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_unprocessed_docs():
    for doc in Document.objects.filter(processed=False):
        file_path = doc.file.path
        ext = os.path.splitext(file_path)[1].lower()

        loader = PyPDFLoader(file_path) if ext == ".pdf" else UnstructuredFileLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        texts = [chunk.page_content for chunk in chunks]
        store = FAISS.from_texts(texts, EMB)
        user_folder = os.path.join("vectorstores", f"user_{doc.user.id}")
        os.makedirs(user_folder, exist_ok=True)
        store.save_local(user_folder)

        doc.processed = True
        doc.save()
        print(f"✅ Processed: {doc.title}")

