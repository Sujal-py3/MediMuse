import os
import pickle
import time

import requests
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together

st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")

# Define layout
col1, col2 = st.columns([1, 3])  # Adjust column width as needed

# Apply CSS styling
st.markdown(
    """
    <style>
    .hospital-title {
        font-family: 'Arial', sans-serif;
        color: #3D7E9A;
        text-align: left;
        margin: 0;
    }
    .hospital-logo {
        display: block;
        margin: 0;
        width: 100%;
    }
    div.stButton > button:first-child {
        background-color: #e0f7fa;
        color: #00796b;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
    }
    div.stButton > button:active {
        background-color: #4db6ac;
        color: #ffffff;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    .chat-title {
        font-family: 'Arial', sans-serif;
        color: #00796b;
        font-size: 24px;
        margin-top: 20px;
        text-align: left;
    }
    .chat-separator {
        margin-top: -10px;
        margin-bottom: 20px;
        border-top: 2px solid #e0f7fa;
    }
    .chat-input {
        font-family: 'Arial', sans-serif;
        padding: 10px;
        border: 2px solid #4db6ac;
        border-radius: 12px;
        width: 100%;
    }
    .chat-container {
        margin-top: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

if "appointments" not in st.session_state:
    st.session_state.appointments = {}  # Store appointments centrally

if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False

# Define a function to reset the conversation and state
def reset_conversation_and_state():
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
    st.session_state.appointments = {}  # Clear appointments when switching roles

# Load FAISS index and initialize the conversational chain
def load_faiss_index():
    index_folder = "index"
    faiss_index_file = os.path.join(index_folder, "index.faiss")
    metadata_file = os.path.join(index_folder, "index.pkl")

    if not os.path.exists(faiss_index_file):
        st.error(f"FAISS index file not found: {faiss_index_file}")
        return None

    if not os.path.exists(metadata_file):
        st.error(f"Metadata file not found: {metadata_file}")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
        )
        db = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return db_retriever
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Define role-specific conversational chains
def create_conversational_chain(retriever, role):
    prompt_template = {
        "Doctor": """<s>[INST]You are an AI assistant specifically trained to assist doctors. Provide precise and accurate responses based solely on the user's query. Avoid generating additional questions or prompts. Your responses should be clear and concise, directly answering the question asked.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
""",
        "Nurse": """<s>[INST]You are an AI assistant specifically trained to assist nurses. Provide clear and practical responses based solely on the user's query. Avoid generating additional questions or prompts. Your responses should be easy to understand and directly answer the question asked.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
""",
        "Patient": """<s>[INST]You are an AI assistant specifically trained to assist patients. Provide simple and direct responses based solely on the user's query. Avoid generating additional questions or prompts. Your responses should be easy to understand and directly answer the question asked.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
    }

    prompt = PromptTemplate(template=prompt_template[role], input_variables=["context", "question", "chat_history"])

    os.environ["TOGETHER_AI"] = "3b01ba7c029199b51dfa32baa6aff8c3d261a60c4552c05dac17b95b2c7bf964"
    TOGETHER_AI_API = os.environ["TOGETHER_AI"]
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_tokens=1024, together_api_key=f"{TOGETHER_AI_API}"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, memory=st.session_state.memory, retriever=retriever, combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa

# Manage appointments
def manage_appointments(input_text):
    response = ""
    if "set appointment" in input_text or "schedule appointment" in input_text:
        date_time = st.text_input("Enter the appointment date and time (e.g., '2024-08-25 10:00 AM'):")
        details = st.text_input("Enter the appointment details (e.g., 'Routine check-up with Dr. Smith'):")
        if st.button("Confirm Appointment"):
            st.session_state.appointments[date_time] = details
            response = f"Appointment scheduled on {date_time} for {details}."
    elif "cancel appointment" in input_text:
        date_time = st.text_input("Enter the appointment date and time to cancel (e.g., '2024-08-25 10:00 AM'):")
        if st.button("Cancel Appointment"):
            if date_time in st.session_state.appointments:
                del st.session_state.appointments[date_time]
                response = f"Appointment on {date_time} has been canceled."
            else:
                response = "No appointment found for the specified time."
    elif "view appointments" in input_text:
        if st.session_state.appointments:
            response = "Your upcoming appointments are:\n"
            for date_time, details in st.session_state.appointments.items():
                response += f"{date_time}: {details}\n"
        else:
            response = "You have no upcoming appointments."
    elif "appointment on" in input_text:
        query_date = input_text.split("on")[-1].strip()
        if query_date in st.session_state.appointments:
            response = f"On {query_date}, you have an appointment for: {st.session_state.appointments[query_date]}"
        else:
            response = f"No appointment found for {query_date}."
    else:
        response = None
    
    return response

# Display chat messages
def display_chat_messages(messages):
    for message in messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

# Process user input
def process_user_input(input_text, qa):
    if st.session_state.user_role == "Nurse":
        appointment_response = manage_appointments(input_text)
        if appointment_response:
            with st.chat_message("assistant"):
                st.write(appointment_response)
            st.session_state.messages.append({"role": "assistant", "content": appointment_response})
            return
    
    with st.chat_message("user"):
        st.write(input_text)

    st.session_state.messages.append({"role": "user", "content": input_text})

    with st.chat_message("assistant"):
        with st.spinner("Processing your query 🩺..."):
            result = qa.invoke(input=input_text)
            answer = result.get("answer", "").strip()
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Role selection page
def role_selection_page():
    st.title("Welcome to the AI Healthcare Assistant")
    role = st.selectbox("Please select your role:", ["Doctor", "Nurse", "Patient"])
    if st.button("Proceed"):
        st.session_state.user_role = role
        st.session_state.page = "chat"
        reset_conversation_and_state()  # Clear previous state

# Main chat page
def chat_page():
    retriever = load_faiss_index()
    if retriever is None:
        return

    qa = create_conversational_chain(retriever, st.session_state.user_role)

    # Left Column (Image and Hospital Name)
    with col1:
        st.markdown('<h1 class="hospital-title">WOODPECKER HOSPITAL 🏥</h1>', unsafe_allow_html=True)
        image_path = os.path.join(os.getcwd(), "healthcare_logo.png")
        st.image(image_path, use_column_width=True, caption="Healthcare Logo")

    # Right Column (Chatbot and Prompt)
    with col2:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chat-title">Ask your Query! 👨‍⚕️👩‍⚕️</h2>', unsafe_allow_html=True)
        st.markdown('<hr class="chat-separator">', unsafe_allow_html=True)

        # Main app code
        for message in st.session_state.messages:
            with st.chat_message(message.get("role")):
                st.write(message.get("content"))

        with st.form(key='chat_form', clear_on_submit=True):
            input_prompt = st.text_input("Type your query here...", key="user_input")
            submit_button = st.form_submit_button(label='Send')

            if submit_button and input_prompt:
                process_user_input(input_prompt, qa)

        st.button("Reset All Chat 🗑️", on_click=reset_conversation_and_state)  # Add reset button
        st.button("Back to Role Selection", on_click=lambda: st.session_state.update({"page": "role_selection"}))  # Back button

        st.markdown('</div>', unsafe_allow_html=True)

# Main app logic
if "page" not in st.session_state:
    st.session_state.page = "role_selection"

if st.session_state.page == "role_selection":
    role_selection_page()
elif st.session_state.page == "chat":
    chat_page()
