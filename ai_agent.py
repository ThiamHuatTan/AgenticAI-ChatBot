# -----------------------------
# Imports
# -----------------------------
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, Dict, Any, List, Tuple
import os
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import re
import tempfile
import requests
import io
from pathlib import Path
import mimetypes

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Enhanced Conversation History Storage with Variable Tracking & File Management
# -----------------------------
class ConversationManager:
    def __init__(self, storage_file="conversations.json"):
        self.storage_file = storage_file
        self.conversations = self._load_conversations()
    
    def _load_conversations(self):
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return {}
    
    def _save_conversations(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "messages": [],
                "variables": {},  # Store variables extracted from conversation
                "context_summary": "",  # Summary of conversation context
                "uploaded_files": []  # Store information about uploaded files
            }
            self._save_conversations()
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            self.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            
            # Extract variables from the message
            self._extract_variables(conversation_id, content, role)
            
            self._save_conversations()
    
    def _extract_variables(self, conversation_id: str, content: str, role: str):
        """Extract variables from messages using pattern matching"""
        if role != "user":
            return
            
        variable_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9]+(?:\.[0-9]+)?)\b'
        matches = re.findall(variable_pattern, content)
        
        for var_name, var_value in matches:
            try:
                if '.' in var_value:
                    value = float(var_value)
                else:
                    value = int(var_value)
                self.conversations[conversation_id]["variables"][var_name] = value
            except ValueError:
                self.conversations[conversation_id]["variables"][var_name] = var_value
    
    def add_uploaded_file(self, conversation_id: str, file_info: Dict[str, Any]):
        """Add information about an uploaded file to the conversation"""
        if conversation_id in self.conversations:
            if "uploaded_files" not in self.conversations[conversation_id]:
                self.conversations[conversation_id]["uploaded_files"] = []
            self.conversations[conversation_id]["uploaded_files"].append(file_info)
            self._save_conversations()
    
    def get_uploaded_files(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get list of uploaded files for the conversation"""
        if conversation_id in self.conversations:
            return self.conversations[conversation_id].get("uploaded_files", [])
        return []
    
    def get_conversation_history(self, conversation_id: str, max_messages: int = 20) -> List[Tuple[str, str]]:
        if conversation_id not in self.conversations:
            return []
        messages = self.conversations[conversation_id]["messages"]
        recent_messages = messages[-max_messages:]
        return [(msg["role"], msg["content"]) for msg in recent_messages]
    
    def get_full_conversation_text(self, conversation_id: str, max_messages: int = 10) -> str:
        if conversation_id not in self.conversations:
            return ""
        messages = self.conversations[conversation_id]["messages"]
        recent_messages = messages[-max_messages:]
        conversation_text = ""
        for msg in recent_messages:
            if msg["role"] == "user":
                conversation_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation_text += f"Assistant: {msg['content']}\n"
        return conversation_text
    
    def get_variables_context(self, conversation_id: str) -> str:
        if conversation_id not in self.conversations or not self.conversations[conversation_id]["variables"]:
            return ""
        variables = self.conversations[conversation_id]["variables"]
        variables_text = "Known variables from this conversation:\n"
        for var_name, var_value in variables.items():
            variables_text += f"{var_name} = {var_value}\n"
        return variables_text
    
    def update_context_summary(self, conversation_id: str, summary: str):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["context_summary"] = summary
            self._save_conversations()
    
    def get_context_summary(self, conversation_id: str) -> str:
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]["context_summary"]
        return ""
    
    def clear_conversation(self, conversation_id: str):
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["messages"] = []
            self.conversations[conversation_id]["variables"] = {}
            self.conversations[conversation_id]["context_summary"] = ""
            self.conversations[conversation_id]["uploaded_files"] = []
            self.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            self._save_conversations()

# Global conversation manager
conversation_manager = ConversationManager()

# -----------------------------
# File Processing Tools - COMPLETELY FIXED VERSION
# -----------------------------
def detect_file_type(filename: str, file_type: str = None) -> str:
    """Detect file type from filename and MIME type"""
    # First, try to use the provided file_type (MIME type)
    if file_type:
        mime_to_extension = {
            'application/pdf': '.pdf',
            'text/plain': '.txt',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'application/vnd.ms-powerpoint': '.ppt',
            'text/csv': '.csv',
        }
        if file_type in mime_to_extension:
            return mime_to_extension[file_type]
    
    # Fall back to file extension
    file_extension = Path(filename).suffix.lower()
    if file_extension:
        return file_extension
    
    # Final fallback
    return '.bin'  # Binary file

def get_file_loader(file_path: str, file_type: str = None):
    """Get appropriate loader based on file type - FIXED VERSION"""
    detected_type = detect_file_type(file_path, file_type)
    print(f"DEBUG: Detected file type: {detected_type} for file: {file_path}")
    
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.csv': CSVLoader,
    }
    
    loader_class = loaders.get(detected_type)
    if loader_class:
        print(f"DEBUG: Using loader: {loader_class.__name__}")
        return loader_class(file_path)
    else:
        # For unknown file types, try text loader as last resort
        print(f"DEBUG: No specific loader found, trying text loader")
        try:
            return TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
        except Exception as e:
            raise ValueError(f"Unsupported file type: {detected_type}. Error: {str(e)}")

def process_uploaded_file(file_content: bytes, filename: str, file_type: str = None) -> Dict[str, Any]:
    """Process uploaded file and extract content - COMPLETELY FIXED VERSION"""
    try:
        print(f"DEBUG: Processing file {filename}, size: {len(file_content)} bytes, provided type: {file_type}")
        
        # Create temporary file with proper permissions
        suffix = Path(filename).suffix or '.tmp'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
            print(f"DEBUG: Created temp file: {tmp_file_path}")
        
        try:
            # Load and process file
            loader = get_file_loader(tmp_file_path, file_type)
            documents = loader.load()
            print(f"DEBUG: Loaded {len(documents)} documents")
            
            # Extract text content
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            content = "\n\n".join([doc.page_content for doc in chunks])
            
            print(f"DEBUG: Extracted content length: {len(content)} characters")
            if content:
                print(f"DEBUG: First 200 chars: {content[:200]}")
            
            # Generate a brief summary of the file content
            summary = generate_file_summary(content, filename)
            
            result = {
                "filename": filename,
                "file_type": file_type or detect_file_type(filename, file_type),
                "content": content,
                "summary": summary,
                "chunk_count": len(chunks),
                "document_count": len(documents),
                "processed_at": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as load_error:
            print(f"DEBUG: Error loading file: {load_error}")
            # Try alternative method for PDFs
            if detect_file_type(filename, file_type) == '.pdf':
                print(f"DEBUG: Trying alternative PDF processing")
                try:
                    # Alternative PDF processing using PyPDF2 directly
                    import PyPDF2
                    with open(tmp_file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        content = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            content += page.extract_text() + "\n\n"
                        
                        if content.strip():
                            result = {
                                "filename": filename,
                                "file_type": '.pdf',
                                "content": content,
                                "summary": generate_file_summary(content, filename),
                                "chunk_count": 1,
                                "document_count": len(pdf_reader.pages),
                                "processed_at": datetime.now().isoformat(),
                                "status": "success_alternative"
                            }
                        else:
                            raise ValueError("No text could be extracted from PDF")
                except Exception as pdf_error:
                    print(f"DEBUG: Alternative PDF processing also failed: {pdf_error}")
                    result = {
                        "filename": filename,
                        "file_type": file_type or detect_file_type(filename, file_type),
                        "error": f"Error processing file: {str(load_error)}",
                        "content": "",
                        "processed_at": datetime.now().isoformat(),
                        "status": "error"
                    }
            else:
                result = {
                    "filename": filename,
                    "file_type": file_type or detect_file_type(filename, file_type),
                    "error": f"Error processing file: {str(load_error)}",
                    "content": "",
                    "processed_at": datetime.now().isoformat(),
                    "status": "error"
                }
        
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
            print(f"DEBUG: Cleaned up temp file: {tmp_file_path}")
        except Exception as cleanup_error:
            print(f"DEBUG: Error cleaning up temp file: {cleanup_error}")
        
        return result
        
    except Exception as e:
        print(f"DEBUG: General error in process_uploaded_file: {e}")
        return {
            "filename": filename,
            "file_type": file_type or detect_file_type(filename, file_type),
            "error": f"Error processing file: {str(e)}",
            "content": "",
            "processed_at": datetime.now().isoformat(),
            "status": "error"
        }

def generate_file_summary(content: str, filename: str, max_length: int = 200) -> str:
    """Generate a brief summary of file content"""
    if not content or not content.strip():
        return "Empty file or no content extracted"
    
    # Simple summary generation
    words = content.split()
    if len(words) > 50:
        # Take first 50 words and last 20 words for summary
        summary_words = words[:50] + ["..."] + words[-20:] if len(words) > 70 else words
        summary = " ".join(summary_words)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
    else:
        summary = content[:max_length] + "..." if len(content) > max_length else content
    
    return summary

# -----------------------------
# Search tool (using Tavily)
# -----------------------------
def get_tavily_search_tool():
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    return TavilySearchResults(tavily_api_key=tavily_api_key)

# -----------------------------
# PDF Reader tool
# -----------------------------
def load_pdf_from_path(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return [doc.page_content for doc in splitter.split_documents(docs)]

def load_pdf_from_url(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp.flush()
            return load_pdf_from_path(tmp.name)
    except Exception as e:
        return [f"Error downloading/reading PDF: {e}"]

def get_pdf_reader_tool():
    return Tool(
        name="PDFReader",
        func=lambda path_or_url: "\n".join(
            load_pdf_from_url(path_or_url) if path_or_url.startswith("http") else load_pdf_from_path(path_or_url)
        ),
        description="Use this to read and extract text from a PDF file given a local path or URL."
    )

# -----------------------------
# File Analysis Tool
# -----------------------------
def get_file_analysis_tool():
    """Tool for analyzing uploaded files"""
    return Tool(
        name="FileAnalyzer",
        func=lambda query: analyze_files_based_on_query(query),
        description="Use this to analyze content of uploaded files. Input should be a query about what to analyze in the files."
    )

def analyze_files_based_on_query(query: str) -> str:
    """Analyze files based on the user's query"""
    try:
        return "File analysis tool activated. Use the available file content in the context to answer questions about uploaded files."
    except Exception as e:
        return f"Error analyzing files: {str(e)}"

# -----------------------------
# Model initialization
# -----------------------------
def initialize_openai_model(model_name: str, **kwargs) -> ChatOpenAI:
    api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise ValueError("OpenAI API key not provided")
    return ChatOpenAI(model_name=model_name, temperature=kwargs.get("temperature", 0), openai_api_key=api_key)

def initialize_groq_model(model_name: str, **kwargs) -> ChatGroq:
    api_key = kwargs.get("api_key", os.getenv("GROQ_API_KEY"))
    if not api_key:
        raise ValueError("Groq API key not provided")
    return ChatGroq(model_name=model_name, temperature=kwargs.get("temperature", 0), groq_api_key=api_key)

MODEL_PROVIDERS = {
    "OpenAI": {"class": initialize_openai_model, "models": ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]},
    "Groq": {"class": initialize_groq_model, "models": ["llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","mixtral-8x7b-32768"]}
}

# -----------------------------
# Enhanced Agent with memory + PDF tool + File analysis + Web Search
# -----------------------------
def get_response_from_ai_agent(
    query: str,
    system_prompt: str,
    model_name: str,
    model_provider: str,
    allow_search: bool,
    conversation_id: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None
):
    try:
        if model_provider not in MODEL_PROVIDERS:
            return {"response": f"Provider '{model_provider}' not supported", "conversation_id": conversation_id, "history_length": 0}
        
        provider_info = MODEL_PROVIDERS[model_provider]
        if model_name not in provider_info["models"]:
            return {"response": f"Model '{model_name}' not supported", "conversation_id": conversation_id, "history_length": 0}
        
        chat_model = provider_info["class"](model_name, **(provider_config or {}))
        
        conversation_id = conversation_manager.create_conversation(conversation_id)
        uploaded_files = conversation_manager.get_uploaded_files(conversation_id)
        
        conversation_manager.add_message(conversation_id, "user", query)
        
        # Build tools based on requirements
        tools = []
        
        # Always add PDF reader tool for URL PDFs
        tools.append(get_pdf_reader_tool())
        
        # Add file analysis tool for uploaded files
        if uploaded_files:
            tools.append(get_file_analysis_tool())
        
        # Add web search tool if allowed
        if allow_search:
            try:
                tools.append(get_tavily_search_tool())
            except Exception as e:
                print(f"Search tool not available: {e}")
        
        # Build file content context for uploaded files
        file_content_section = ""
        if uploaded_files:
            file_content_section = "\n\n=== UPLOADED FILES CONTENT ===\n"
            for file_info in uploaded_files:
                file_content_section += f"\n--- FILE: {file_info['filename']} (Status: {file_info.get('status', 'unknown')}) ---\n"
                if 'content' in file_info and file_info['content'] and file_info['content'].strip():
                    actual_content = file_info['content'].strip()
                    content_preview = actual_content[:4000]
                    file_content_section += f"CONTENT:\n{content_preview}\n"
                    if len(actual_content) > 4000:
                        file_content_section += f"... (showing first 4000 of {len(actual_content)} characters)\n"
                elif 'error' in file_info:
                    file_content_section += f"ERROR: {file_info['error']}\n"
                else:
                    file_content_section += "No content extracted or empty file\n"
                file_content_section += f"--- END OF FILE: {file_info['filename']} ---\n"
            file_content_section += "\n=== END UPLOADED FILES CONTENT ===\n"
        
        # Get conversation context
        conversation_context = conversation_manager.get_full_conversation_text(conversation_id)
        variables_context = conversation_manager.get_variables_context(conversation_id)
        context_summary = conversation_manager.get_context_summary(conversation_id)
        
        # Build context parts
        context_parts = []
        if context_summary:
            context_parts.append(f"Conversation summary: {context_summary}")
        if conversation_context:
            context_parts.append(f"Recent conversation:\n{conversation_context}")
        if variables_context:
            context_parts.append(variables_context)
        if file_content_section:
            context_parts.append(f"Uploaded files content:\n{file_content_section}")
        
        full_context = "\n\n".join(context_parts) if context_parts else ""
        
        # Enhanced system prompt
        enhanced_system_prompt = f"""{system_prompt}

CAPABILITIES:
- You can analyze UPLOADED FILES (content provided above)
- You can read PDFs from URLs using the PDFReader tool
- You can search the web using the TavilySearch tool (if enabled)
- You can analyze uploaded file content directly

TOOLS AVAILABLE: {[tool.name for tool in tools]}

INSTRUCTIONS:
1. For uploaded files: Analyze the content provided above
2. For PDF URLs: Use the PDFReader tool to read the content
3. For web searches: Use the TavilySearch tool to find information
4. Combine information from multiple sources when needed"""
        
        # Use agent if we have tools, otherwise use direct chat
        if tools:
            try:
                agent = initialize_agent(
                    tools=tools,
                    llm=chat_model,
                    agent="chat-conversational-react-description",
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=5
                )
                
                # Prepare agent input
                if full_context:
                    agent_input = f"""{enhanced_system_prompt}

{full_context}

Current query: {query}

Please use the appropriate tools to handle this request."""
                else:
                    agent_input = f"{enhanced_system_prompt}\n\nCurrent query: {query}"
                
                chat_history = conversation_manager.get_conversation_history(conversation_id)
                formatted_history = [{"role": role, "content": content} for role, content in chat_history]
                
                result = agent.invoke(input={"input": agent_input, "chat_history": formatted_history})
                
                if isinstance(result, dict):
                    response = result.get("output", "No response from agent.")
                else:
                    response = str(result.content) if hasattr(result, "content") else str(result)
                    
            except Exception as agent_error:
                print(f"Agent error: {agent_error}, falling back to direct chat")
                # Fallback to direct chat
                response = handle_direct_chat(
                    chat_model, enhanced_system_prompt, full_context, query, 
                    conversation_manager, conversation_id
                )
        else:
            # No tools available, use direct chat
            response = handle_direct_chat(
                chat_model, enhanced_system_prompt, full_context, query,
                conversation_manager, conversation_id
            )
        
        conversation_manager.add_message(conversation_id, "assistant", response)
        
        if len(conversation_manager.get_conversation_history(conversation_id)) % 5 == 0:
            summary = f"Conversation summary. Variables: {conversation_manager.get_variables_context(conversation_id)}. Files: {len(uploaded_files)}"
            conversation_manager.update_context_summary(conversation_id, summary)
        
        return {
            "response": response, 
            "conversation_id": conversation_id, 
            "history_length": len(conversation_manager.get_conversation_history(conversation_id)),
            "file_count": len(uploaded_files)
        }
    
    except Exception as e:
        return {"response": f"Error calling AI agent: {e}", "conversation_id": conversation_id, "history_length": 0}

def handle_direct_chat(chat_model, system_prompt: str, context: str, query: str, 
                      conversation_manager, conversation_id: str) -> str:
    """Handle direct chat without agent tools"""
    messages = [{"role": "system", "content": system_prompt}]
    
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    
    # Add conversation history
    chat_history = conversation_manager.get_conversation_history(conversation_id)
    for role, content in chat_history:
        messages.append({"role": role, "content": content})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    result = chat_model.invoke(messages)
    return str(result.content) if hasattr(result, "content") else str(result)

# -----------------------------
# File Upload Function with better debugging
# -----------------------------
def handle_file_upload(file_content: bytes, filename: str, conversation_id: str, file_type: str = None) -> Dict[str, Any]:
    """Handle file upload and processing - FIXED VERSION"""
    try:
        print(f"DEBUG: Starting file upload for {filename}, size: {len(file_content)} bytes")
        
        # Ensure conversation exists
        conversation_id = conversation_manager.create_conversation(conversation_id)
        print(f"DEBUG: Using conversation ID: {conversation_id}")
        
        # Process the file
        file_info = process_uploaded_file(file_content, filename, file_type)
        
        # Add to conversation
        conversation_manager.add_uploaded_file(conversation_id, file_info)
        
        # Verify the file was actually added
        current_files = conversation_manager.get_uploaded_files(conversation_id)
        print(f"DEBUG: After upload, conversation has {len(current_files)} files")
        
        # Build response
        if file_info.get('status') == 'error' or 'error' in file_info:
            response = {
                "success": False,
                "file_info": file_info,
                "message": f"Error processing file '{filename}': {file_info.get('error', 'Unknown error')}"
            }
        else:
            content_length = len(file_info.get('content', ''))
            response = {
                "success": True,
                "file_info": file_info,
                "message": f"File '{filename}' uploaded and processed successfully. Content length: {content_length} characters. Pages/chunks: {file_info.get('chunk_count', 0)}. Status: {file_info.get('status', 'success')}"
            }
        
        print(f"DEBUG: Upload response: {response['message']}")
        return response
        
    except Exception as e:
        print(f"DEBUG: Exception in handle_file_upload: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Error uploading file: {str(e)}"
        }

# -----------------------------
# Helper functions
# -----------------------------
def get_available_providers():
    return {provider: {"models": info["models"], "requires_api_key": True} for provider, info in MODEL_PROVIDERS.items()}

def clear_conversation_history(conversation_id: str):
    conversation_manager.clear_conversation(conversation_id)
    return {"status": "success", "message": f"Conversation {conversation_id} cleared"}

def get_conversation_variables(conversation_id: str):
    if conversation_id in conversation_manager.conversations:
        return conversation_manager.conversations[conversation_id]["variables"]
    return {}

def get_conversation_files(conversation_id: str):
    return conversation_manager.get_uploaded_files(conversation_id)