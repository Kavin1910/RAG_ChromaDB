import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import os

# Streamlit Configuration
st.set_page_config(page_title="MyAyur Health Chatbot", page_icon="🌿", layout="wide")

# Configuration Constants
LOCAL_MODEL = "llama3.1:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"
PERSIST_DIRECTORY = 'vector_store'

# Function to load and split documents
def load_and_split_documents(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Function to create vector store
def create_vector_store(documents, embedding_model):
    return Chroma.from_documents(
        documents=documents, 
        embedding=OllamaEmbeddings(model=embedding_model),
        persist_directory=PERSIST_DIRECTORY
    )

# Function to create RAG chain
def create_rag_chain(llm, retriever):
    template = """You are a professional healthcare assistant focused on Ayurvedic knowledge.
    
    Context: {context}
    Chat History: {history}
    
    User Question: {question}
    Helpful Response:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template
    )
    
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory
        }
    )

# Streamlit App
def main():
    st.title("🌿 MyAyur Health Chatbot")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Document Upload Section
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    
    if uploaded_file is not None:
        with st.spinner('Processing document...'):
            # Save uploaded file
            os.makedirs("uploaded_docs", exist_ok=True)
            file_path = os.path.join("uploaded_docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load and split documents
            documents = load_and_split_documents(file_path)
            
            # Create vector store
            vectorstore = create_vector_store(documents, EMBEDDING_MODEL)
            retriever = vectorstore.as_retriever()
            
            # Initialize LLM
            llm = Ollama(
                base_url="http://localhost:11434",
                model=LOCAL_MODEL,
                verbose=True
            )
            
            # Create RAG chain
            qa_chain = create_rag_chain(llm, retriever)
            
            # Chat input
            user_query = st.chat_input("Ask a question about your document")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Display chat history
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Generate and stream assistant response
                with st.chat_message("assistant"):
                    # Use st.empty() for streaming
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    # Perform RAG query
                    try:
                        # Use invoke instead of stream if streaming is not working
                        response = qa_chain.invoke({"query": user_query})
                        
                        # Extract the response text
                        if isinstance(response, dict):
                            response_text = response.get('result', '')
                        else:
                            response_text = str(response)
                        
                        # Update placeholder with full response
                        response_placeholder.markdown(response_text)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response_text
                        })
                    
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
    
    else:
        st.warning("Please upload a PDF document to start chatting.")

if __name__ == "__main__":
    main()