import streamlit as st
import requests
import time
import uuid
import io

# -----------------------------
# Streamlit app setup
# -----------------------------
st.set_page_config(page_title="Multi-Provider AI Agent", layout="centered")
st.header("Multi-Provider AI Chatbot Application")
st.write("Create and Interact with AI Agents from multiple providers! Upload files and ask questions about them!")

# Initialize session state for conversation management
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Helper function for fallback providers
def get_fallback_providers():
    return {
        "OpenAI": {
            "models": ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
            "requires_api_key": True
        },
        "Groq": {
            "models":  ["llama-3.3-70b-versatile","deepseek-r1-distill-llama-70b","mixtral-8x7b-32768"],
            "requires_api_key": True
        }
    }

def fetch_providers_with_retry(url, max_retries=2, timeout=5):
    """Try to fetch providers with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json(), True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
                continue
            return None, False
    return None, False

# -----------------------------
# Initialize session state for providers
# -----------------------------
if 'providers' not in st.session_state:
    providers_data, success = fetch_providers_with_retry("http://127.0.0.1:9999/providers")
    
    if success:
        st.session_state.providers = providers_data
        st.session_state.backend_available = True
    else:
        st.warning("Could not connect to backend. Using fallback providers.")
        st.session_state.backend_available = False
        st.session_state.providers = get_fallback_providers()

# Sidebar
st.sidebar.title("Multi-Provider AI Chatbot")
st.sidebar.image(
    "https://miro.medium.com/v2/resize:fit:1400/1*hdd2IGtXs3E8rsa98m-kCg.png",
    caption="Your Multi-Provider AI Assistant",
    use_container_width=True
)

# Show connection status in sidebar
if st.session_state.get('backend_available', False):
    st.sidebar.success("✓ Backend connected")
else:
    st.sidebar.error("✗ Backend not connected")

# Conversation management in sidebar
st.sidebar.title("Conversation Management")
if st.session_state.conversation_id:
    st.sidebar.info(f"Conversation ID: {st.session_state.conversation_id[:8]}...")
    st.sidebar.write(f"Messages: {len(st.session_state.conversation_history)}")
    st.sidebar.write(f"Files: {len(st.session_state.uploaded_files)}")
    
    if st.sidebar.button("🗑️ Clear Conversation", key="clear_conv_btn"):
        if st.session_state.backend_available:
            try:
                response = requests.post(
                    "http://127.0.0.1:9999/clear_conversation",
                    json={"conversation_id": st.session_state.conversation_id}
                )
                if response.status_code == 200:
                    st.session_state.conversation_history = []
                    st.session_state.uploaded_files = []
                    st.sidebar.success("Conversation cleared!")
                else:
                    st.sidebar.error("Failed to clear conversation")
            except:
                st.session_state.conversation_history = []
                st.session_state.uploaded_files = []
                st.sidebar.success("Conversation cleared locally!")
        else:
            st.session_state.conversation_history = []
            st.session_state.uploaded_files = []
            st.sidebar.success("Conversation cleared locally!")
        
        st.session_state.conversation_id = None
        st.rerun()
else:
    st.sidebar.info("No active conversation")

# Use a checkbox for show/hide history in the main area
show_history = st.sidebar.checkbox(
    "📋 Show Conversation History", 
    value=st.session_state.show_history,
    key="show_history_checkbox"
)

# Update the session state based on checkbox
st.session_state.show_history = show_history

# File upload section in sidebar
st.sidebar.title("File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload files to analyze",
    type=['pdf', 'txt', 'docx', 'doc', 'pptx', 'ppt', 'csv'],
    help="Supported formats: PDF, TXT, DOCX, PPTX, CSV"
)

# Initialize processing state
if 'processing_file' not in st.session_state:
    st.session_state.processing_file = False

if uploaded_file is not None and st.session_state.backend_available:
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if st.sidebar.button("Process Uploaded File"):
        st.session_state.processing_file = True

# Process file if the state is set
if st.session_state.processing_file and uploaded_file is not None:
    # Show processing message in main area since sidebar spinner doesn't exist
    processing_placeholder = st.empty()
    with processing_placeholder.container():
        st.info("🔄 Processing uploaded file...")
    
    try:
        # Prepare file for upload
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {
            'conversation_id': st.session_state.conversation_id
        }
        
        response = requests.post(
            "http://127.0.0.1:9999/upload_file",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                st.sidebar.success(f"✅ {result['message']}")
                # Refresh file list
                try:
                    files_response = requests.post(
                        "http://127.0.0.1:9999/get_files",
                        json={"conversation_id": st.session_state.conversation_id}
                    )
                    if files_response.status_code == 200:
                        st.session_state.uploaded_files = files_response.json().get('files', [])
                except:
                    pass
            else:
                st.sidebar.error(f"❌ {result.get('message', 'Unknown error')}")
        else:
            st.sidebar.error("Failed to upload file")
    except Exception as e:
        st.sidebar.error(f"Error uploading file: {str(e)}")
    
    # Clear processing state and placeholder
    st.session_state.processing_file = False
    processing_placeholder.empty()

# Show uploaded files in sidebar
if st.session_state.uploaded_files:
    st.sidebar.title("Uploaded Files")
    for file_info in st.session_state.uploaded_files:
        with st.sidebar.expander(f"📄 {file_info.get('filename', 'Unknown')}"):
            st.write(f"Type: {file_info.get('file_type', 'Unknown')}")
            if 'summary' in file_info:
                st.write(f"Summary: {file_info['summary']}")
            if 'error' in file_info:
                st.error(f"Error: {file_info['error']}")

st.sidebar.title("Instructions 📜")
st.sidebar.markdown(
    """
    1. Select your AI Provider (OpenAI or Groq).
    2. Select the Model for your chosen provider.
    3. Enter your Persona/Role for AI agent.
    4. Choose if you want to allow web search (requires Tavily API key).
    5. Upload files (optional) - the AI can analyze them.
    6. Enter your Query and ask Agent!
    """
)

# -----------------------------
# Input fields
# -----------------------------
# Provider selection - Set Groq as default
available_providers = list(st.session_state.providers.keys()) if st.session_state.providers else ["OpenAI", "Groq"]

# Set Groq as default by finding its index
default_index = 0
if "Groq" in available_providers:
    default_index = available_providers.index("Groq")

provider = st.radio(
    "Select Provider:", 
    available_providers,
    index=default_index  # This sets Groq as default
)

# Model selection based on provider
if provider and st.session_state.providers:
    provider_info = st.session_state.providers.get(provider, {})
    available_models = provider_info.get("models", [])
    
    # Set default model based on provider
    default_model_index = 0
    if provider == "Groq" and available_models:
        # Prefer "llama-3.3-70b-versatile" for Groq if available
        preferred_models = ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "mixtral-8x7b-32768"]
        for preferred_model in preferred_models:
            if preferred_model in available_models:
                default_model_index = available_models.index(preferred_model)
                break
    
    selected_model = st.selectbox(
        f"Select {provider} Model:", 
        available_models,
        index=default_model_index
    )
else:
    selected_model = st.selectbox("Select Model:", ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"])

# Provider-specific configuration
provider_config = {}

if provider == "OpenAI":
    api_key = st.text_input("OpenAI API Key (optional):", type="password", 
                           help="If not provided, will use OPENAI_API_KEY environment variable")
    if api_key:
        provider_config["api_key"] = api_key
        
elif provider == "Groq":
    api_key = st.text_input("Groq API Key (optional):", type="password",
                           help="If not provided, will use GROQ_API_KEY environment variable")
    if api_key:
        provider_config["api_key"] = api_key

# Common inputs
system_prompt = st.text_area(
    "Define your AI Agent:",
    height=70,
    placeholder="Type your system prompt here...",
    help="Example: 'You are a helpful assistant that can analyze documents and answer questions about them.'"
)

allow_web_search = st.checkbox("Allow Web Search (requires Tavily API key)")

# File upload info
if st.session_state.uploaded_files:
    st.info(f"📁 {len(st.session_state.uploaded_files)} file(s) uploaded and ready for analysis")

user_query = st.text_area(
    "Enter your query:", 
    height=150, 
    placeholder="Ask Anything! You can ask questions about uploaded files, request analysis, or general questions.",
    help="Example: 'What are the main points in the uploaded document?' or 'Summarize the key findings from my files.'"
)

API_URL = "http://127.0.0.1:9999/chat"

# -----------------------------
# Ask Agent button
# -----------------------------
if st.button("Ask Agent!"):
    if not user_query.strip():
        st.warning("Please enter a query before asking the agent!")
    else:
        # Ensure we have a conversation ID
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt or "You are a helpful assistant that can analyze documents and answer questions.",
            "query": user_query,
            "allow_search": allow_web_search,
            "conversation_id": st.session_state.conversation_id,
            "provider_config": provider_config
        }

        try:
            with st.spinner("Waiting for agent response..."):
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                response_data = response.json()

            if "error" in response_data:
                st.error(response_data["error"])
            else:
                # Update conversation state
                st.session_state.conversation_id = response_data.get("conversation_id")
                
                # Add to conversation history
                current_time = time.strftime("%H:%M:%S")
                st.session_state.conversation_history.append(("user", user_query, current_time))
                st.session_state.conversation_history.append(("assistant", response_data.get("response", ""), current_time))
                
                st.subheader("Agent Response")
                st.markdown(
                    f"**Final Response:** {response_data.get('response', response_data)}"
                )
                
                # Show conversation info
                st.info(f"Conversation ID: {st.session_state.conversation_id} | Messages: {response_data.get('history_length', 0)} | Files: {len(st.session_state.uploaded_files)}")

        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to backend. Make sure your backend is running on port 9999.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The backend might be busy or unresponsive.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Display current conversation in main area if there's history AND history is enabled
if st.session_state.show_history and st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("Conversation History")
    
    for i, (role, message, timestamp) in enumerate(st.session_state.conversation_history):
        if role == "user":
            with st.chat_message("user"):
                st.write(message)
                st.caption(f"You • {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.write(message)
                st.caption(f"AI • {timestamp}")
elif st.session_state.show_history and not st.session_state.conversation_history:
    st.info("No conversation history yet. Start a conversation above!")

# Display file analysis examples if files are uploaded
if st.session_state.uploaded_files and not st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("💡 Try asking questions like:")
    
    example_questions = [
        "What are the main points in the uploaded documents?",
        "Can you summarize the key information from my files?",
        "Compare and contrast the information across different files",
        "Extract important data points or statistics from the documents",
        "What are the key recommendations or findings in these files?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            # Auto-fill the query with the example question
            user_query = question
            st.rerun()

# Add some styling
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .uploadedFile {
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)