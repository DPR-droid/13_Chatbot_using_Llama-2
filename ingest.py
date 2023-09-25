
# If to run by GPU (cuda)
# https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud
# nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
# Cuda compilation tools, release 11.7, V11.7.64
# Build cuda_11.7.r11.7/compiler.31294372_0
# Select your preferences and run the install command.
# https://pytorch.org/get-started/locally/ 
# conda GPU
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# or pip GPU
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# or CPU
# pip3 install torch torchvision torchaudio

# conda install pypdf
# pip install sentence_transformers
# conda install faiss-gpu
# conda install faiss-cpu


# Import necessary modules

# https://docs.langchain.com/docs/components/indexing/text-splitters
# Often times you want to split large text documents into smaller chunks to better work with language models.
from langchain.text_splitter import RecursiveCharacterTextSplitter 
# https://python.langchain.com/docs/modules/data_connection/document_loaders/
# Use document loaders to load data from a source as Document's
from langchain.document_loaders import PyPDFLoader, DirectoryLoader

# https://python.langchain.com/docs/modules/data_connection/text_embedding/
# The Embeddings class is a class designed for interfacing with text embedding models.
from langchain.embeddings import HuggingFaceEmbeddings 

# https://python.langchain.com/docs/modules/data_connection/vectorstores/
# One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, 
# and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query.
from langchain.vectorstores import FAISS 

# Check Path C:/{path}
DATA_PATH = "PDF/"
DB_FAISS_PATH = "db/"

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

    texts = text_splitter.split_documents(documents)

    # Select to use CPU/GPU
    # embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cpu'})
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device': 'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)


# Execute the main function if the script is run as the main program
if __name__ == "__main__":
    create_vector_db()
