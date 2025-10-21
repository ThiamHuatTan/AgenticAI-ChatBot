import streamlit as st
import requests
import time
import uuid
import io
import os
from dotenv import load_dotenv
import json
import tempfile
import pypdf
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain import text_splitters
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Load environment variables
load_dotenv()

# Initialize session state
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'providers' not in st.session_state:
    st.session_state.providers = {
        "OpenAI": {
            "models": ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
            "requires_api_key": True
        },
        "Groq": {
            "models": ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"],
            "requires_api_key": True
        }
    }

# File processing functions (from ai_agent.py)
def process_uploaded_file(file_content: bytes, filename: str):
    """Process uploaded file and extract content"""
    try:
        # Create temporary file
        suffix = Path(filename).suffix or '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Simple text extraction for common formats
            if filename.lower().endswith('.pdf'):
                content = extract_text_from_pdf(tmp_file_path)
            elif filename.lower().endswith('.txt'):
                with open(tmp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                # For other formats, try to read as text
                try:
                    with open(tmp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    content = f"Unsupported file format: {filename}"
            
            # Generate summary
            summary = generate_file_summary(content, filename)
            
            result = {
                "filename": filename,
                "content": content,
                "summary": summary,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success"
            }
            
        except Exception as e:
            result = {
                "filename": filename,
                "error": str(e),
                "status": "error"
            }
        
        # Cleanup
        try:
            os.unlink(tmp_file_path)
        except:
            pass
            
        return result
        
    except Exception as e:
        return {
            "filename": filename,
            "error": str(e),
            "status": "error"
        }

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def generate_file_summary(content: str, filename: str, max_length: int = 200) -> str:
    """Generate a brief summary of file content"""
    if not content or len(content.strip()) == 0:
        return "Empty file or no content extracted"
    
    words = content.split()
    if len(words) > 50:
        summary_words = words[:50] + ["..."] + words[-20:] if len(words) > 70 else words
        summary = " ".join(summary_words)
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    else:
        return content[:max_length] + "..." if len(content) > max_length else content

# AI response function
def get_ai_response(query, system_prompt, model_name, model_provider, conversation_history, uploaded_files_content):
    """Get response from AI model"""
    try:
        # Prepare context from uploaded files
        files_context = ""
        if uploaded_files_content:
            files_context = "\n\nUPLOADED FILES CONTEXT:\n"
            for file_info in uploaded_files_content:
                if file_info.get('content'):
                    files_context += f"\n--- {file_info['filename']} ---\n"
                    files_context += file_info['content'][:2000] + "\n"
        
        # Prepare conversation history
        history_context = ""
        if conversation_history:
            history_context = "\n\nCONVERSATION HISTORY:\n"
            for role, msg, _ in conversation_history[-6:]:  # Last 6 messages
                history_context += f"{role.upper()}: {msg}\n"
        
        # Build full prompt
        full_prompt = f"""{system_prompt}

{files_context}
{history_context}

Current query: {query}

Please respond helpfully and accurately."""
        
        # Initialize the appropriate model
        if model_provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables."
            llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
        elif model_provider == "Groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return "Groq API key not found. Please set GROQ_API_KEY in your environment variables."
            llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
        else:
            return f"Unsupported provider: {model_provider}"
        
        # Get response
        response = llm.invoke(full_prompt)
        return response.content
        
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="AI Document Analyzer", layout="centered")
st.title("🤖 AI Document Analyzer")
st.write("Upload documents and chat with AI about their content!")

# Sidebar
st.sidebar.title("Configuration")

# Provider selection
available_providers = list(st.session_state.providers.keys())
default_index = available_providers.index("Groq") if "Groq" in available_providers else 0
provider = st.sidebar.radio("AI Provider", available_providers, index=default_index)

# Model selection
if provider in st.session_state.providers:
    available_models = st.session_state.providers[provider]["models"]
    default_model_index = 0
    if provider == "Groq":
        preferred_models = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"]
        for pref_model in preferred_models:
            if pref_model in available_models:
                default_model_index = available_models.index(pref_model)
                break
    model_name = st.sidebar.selectbox("Model", available_models, index=default_model_index)

# API Key inputs
if provider == "OpenAI":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                     value=os.getenv("OPENAI_API_KEY", ""))
    os.environ["OPENAI_API_KEY"] = openai_key
elif provider == "Groq":
    groq_key = st.sidebar.text_input("Groq API Key", type="password",
                                   value=os.getenv("GROQ_API_KEY", ""))
    os.environ["GROQ_API_KEY"] = groq_key

# File upload
st.sidebar.title("Document Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose files to analyze",
    type=['pdf', 'txt', 'docx', 'doc', 'pptx', 'csv'],
    help="Upload documents for AI analysis"
)

if uploaded_file and st.sidebar.button("Process File"):
    with st.spinner("Processing file..."):
        file_content = uploaded_file.getvalue()
        result = process_uploaded_file(file_content, uploaded_file.name)
        
        if result.get('status') == 'success':
            st.session_state.uploaded_files.append(result)
            st.sidebar.success(f"✅ {uploaded_file.name} processed successfully!")
        else:
            st.sidebar.error(f"❌ Failed to process {uploaded_file.name}")

# Show uploaded files
if st.session_state.uploaded_files:
    st.sidebar.title("Uploaded Files")
    for file_info in st.session_state.uploaded_files:
        with st.sidebar.expander(f"📄 {file_info['filename']}"):
            if 'summary' in file_info:
                st.write(f"Summary: {file_info['summary']}")
            if 'error' in file_info:
                st.error(f"Error: {file_info['error']}")

# Main chat interface
st.sidebar.title("Chat Settings")
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a helpful AI assistant that analyzes documents and answers questions based on their content.",
    height=100
)

# Conversation management
if st.session_state.conversation_id:
    st.sidebar.info(f"Conversation: {str(st.session_state.conversation_id)[:8]}...")
    if st.sidebar.button("Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.conversation_id = None
        st.rerun()

# User input
user_query = st.text_area(
    "Your Question",
    height=100,
    placeholder="Ask about your uploaded documents or anything else...",
    help="You can reference content from uploaded files in your questions"
)

if st.button("Ask AI") and user_query.strip():
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    # Add user message to history
    current_time = time.strftime("%H:%M:%S")
    st.session_state.conversation_history.append(("user", user_query, current_time))
    
    # Get AI response
    with st.spinner("Analyzing documents and generating response..."):
        response = get_ai_response(
            query=user_query,
            system_prompt=system_prompt,
            model_name=model_name,
            model_provider=provider,
            conversation_history=st.session_state.conversation_history,
            uploaded_files_content=st.session_state.uploaded_files
        )
    
    # Add AI response to history
    st.session_state.conversation_history.append(("assistant", response, current_time))
    
    # Display response
    st.subheader("AI Response")
    st.write(response)

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("Conversation History")
    
    for role, message, timestamp in st.session_state.conversation_history:
        if role == "user":
            with st.chat_message("user"):
                st.write(message)
                st.caption(f"You • {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.write(message)
                st.caption(f"AI • {timestamp}")

# File analysis suggestions
if st.session_state.uploaded_files and not st.session_state.conversation_history:
    st.info("💡 Try asking questions like:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Summarize the main points"):
            st.session_state.conversation_history = []
            st.rerun()
    with col2:
        if st.button("What are the key findings?"):
            st.session_state.conversation_history = []
            st.rerun()