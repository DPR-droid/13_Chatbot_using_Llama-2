# Import necessary modules
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
import os 
from constants import CHROMA_SETTINGS

# Define the directory where the results will be persisted
persist_directory = "db"

# Define the main function
def main():
    # Walk through the "docs" directory and process PDF files
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing {file}")
                # Initialize the PDF loader
                loader = PyPDFLoader(os.path.join(root, file))
    
    # Load documents from the PDF files
    documents = loader.load()
    print("Splitting documents into chunks")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # Split documents into chunks
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create a vector store
    print(f"Creating embeddings. This may take some time...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    
    # Persist the results
    db.persist()
    
    # Clean up and finalize
    db = None 

    print("Ingestion complete! You can now run privateGPT.py to query your documents")

# Execute the main function if the script is run as the main program
if __name__ == "__main__":
    main()
