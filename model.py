from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """<|system|>You are a helpful medical chatbot that provides accurate information based on the given context.</|system|>
<|user|>Use this context to answer the question:
{context}

Question: {question}</|user|>
<|assistant|>I'll provide a clear and concise answer based on the context provided.</|assistant|>"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(
            search_kwargs={
                'k': 1,  # Reduced number of chunks to retrieve
                'score_threshold': 0.5  # Only retrieve relevant chunks
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': prompt,
            'document_separator': '\n',  # Simplified document separation
            'verbose': False  # Reduce logging overhead
        }
    )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",   # Lightweight 1.1B model
        model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", # Quantized version
        model_type="llama",
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        threads=4,
        gpu_layers=0,        # CPU only
        batch_size=8,
        context_length=512,
        stop=["</s>", "[/INST]"],  # Stop tokens for chat format
        stream=True
    )
    return llm

# Cache for embeddings model and db
_embeddings_cache = None
_db_cache = None

#QA Model Function
def qa_bot():
    global _embeddings_cache, _db_cache
    
    # Use cached embeddings if available
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder="./cache"  # Move cache_folder to top level
        )
    
    # Use cached DB if available
    if _db_cache is None:
        _db_cache = FAISS.load_local(DB_FAISS_PATH, _embeddings_cache, allow_dangerous_deserialization=True)
    
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, _db_cache)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Metabolical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="Error: Chain not initialized. Please restart the chat.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    
    try:
        # Call the chain with the message content
        res = await chain.ainvoke(
            {"query": message.content},
            config={"callbacks": [cb]}
        )
        
        # Extract and format the response
        answer = res.get("result", "No answer generated")
        sources = res.get("source_documents", [])
        
        # Format the sources in a cleaner way
        if sources:
            source_text = "\n\nSources:"
            for i, source in enumerate(sources, 1):
                doc_name = source.metadata.get('source', '').split('/')[-1]
                page = source.metadata.get('page', '')
                source_text += f"\n{i}. {doc_name}"
                if page:
                    source_text += f" (Page {page})"
            answer += source_text
        
        await cl.Message(content=answer).send()
    
    except Exception as e:
        error_msg = f"Error processing your request: {str(e)}"
        await cl.Message(content=error_msg).send()

