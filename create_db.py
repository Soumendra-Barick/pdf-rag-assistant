# load pdf
# split into chunks
# create the embeddings
# store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

data = PyPDFLoader("Document loaders/deeplearning.pdf")
docs = data.load()

spliter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200
)

chunks = spliter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma-db"
)
