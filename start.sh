#!/bin/bash

# Create necessary directories
mkdir -p /app/data
mkdir -p /app/vectorstore/db_faiss
mkdir -p /app/cache

# Run the ingest script to create vector store
echo "Initializing vector store..."
python ingest.py

# Start the application
echo "Starting the application..."
exec chainlit run model.py --host 0.0.0.0 --port 10000
