# import logging
# import requests
# import streamlit as st

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Replace with your EC2 instance's public IP or domain
# EC2_ENDPOINT = "https://18.234.176.247/"

# st.title("RAG Query System")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("What would you like to know?"):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     try:
#         with st.spinner('Generating response... This may take a while.'):
#             response = requests.post(f"{EC2_ENDPOINT}/query", json={"text": prompt}, timeout=900)
        
#         if response.status_code == 200:
#             data = response.json()
#             answer = data["response"]
#             debug_info = data["debug_info"]["retrieved_context"]
            
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
#                 with st.expander("View retrieved context"):
#                     st.text(debug_info)
            
#             st.session_state.messages.append({"role": "assistant", "content": answer})
#         else:
#             st.error(f"Error: {response.status_code} - {response.text}")
#     except requests.exceptions.RequestException as e:
#         st.error(f"An error occurred: {str(e)}")

# # Add a button to clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.messages = []
#     st.experimental_rerun()

import logging
import requests
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("PDF Query System")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the healthcare data?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner('Generating response... This may take a while.'):
            response = requests.post("http://localhost:8000/query", json={"text": prompt}, timeout=900)
        
        if response.status_code == 200:
            data = response.json()
            answer = data["response"]
            debug_info = data["debug_info"]["retrieved_context"]
            
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("View retrieved context"):
                    st.text(debug_info)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {str(e)}")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()