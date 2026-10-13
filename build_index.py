import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = DirectoryLoader("D:/PythonProject/paper", glob="**/*.txt",
                         loader_cls=TextLoader,
                         loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"已存入 {len(chunks)} 个 chunk 到 Chroma")