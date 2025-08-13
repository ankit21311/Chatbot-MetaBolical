from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    print("Loading PDF files from directory...")
    import os
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")
    
    documents = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        loader = PyPDFLoader(os.path.join(DATA_PATH, pdf_file))
        documents.extend(loader.load())
        print(f"Finished processing {pdf_file}")
    
    print(f"Loaded {len(documents)} pages total")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("Creating FAISS database (this may take a while)...")
    batch_size = 5000  # Process 5000 documents at a time
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        print(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}...")
        
        # Process first batch
        if i == 0:
            db = FAISS.from_documents(texts[i:end_idx], embeddings)
        # Merge subsequent batches
        else:
            db_batch = FAISS.from_documents(texts[i:end_idx], embeddings)
            db.merge_from(db_batch)
            
        # Save after each batch as checkpoint
        print(f"Saving progress...")
        db.save_local(DB_FAISS_PATH)
        
    print("Done! Vector store has been created successfully.")

if __name__ == "__main__":
    create_vector_db()

