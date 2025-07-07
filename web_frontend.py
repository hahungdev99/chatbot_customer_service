import streamlit as st
import requests
import uuid  # For generating unique session IDs
import json

# Define the backend API URL
BACKEND_URL = "http://127.0.0.1:5000/chat"

# Streamlit app title
st.title("Shop giày BQ")

# Generate a unique session ID (ussid) for each user
if "ussid" not in st.session_state:
    st.session_state.ussid = str(uuid.uuid4())

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Custom CSS for dynamic loading animation
st.markdown("""
<style>
/* Spinner animation */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Dots animation */
@keyframes dots {
    0%, 20% { opacity: 0; }
    50% { opacity: 1; }
    80%, 100% { opacity: 0; }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0, 0, 0, 0.1);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.loading-dots {
    display: flex;
    gap: 5px;
}
.loading-dots span {
    display: block;
    width: 10px;
    height: 10px;
    background-color: #000;
    border-radius: 50%;
    animation: dots 1.4s infinite ease-in-out;
}
.loading-dots span:nth-child(1) { animation-delay: 0s; }
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create a placeholder for the bot's response
    bot_response_placeholder = st.empty()

    # Show a dynamic loading animation while waiting for the response
    bot_response_placeholder.markdown("""
    <div style="display: flex; align-items: center;">
        <div class="loading-spinner"></div>
        <span style="margin-left: 10px;">Đợi xíu nhaaaaa ...</span>
    </div>
    """, unsafe_allow_html=True)

    full_bot_response = ""

    # Send the user input to the backend API
    try:
        with requests.post(
            BACKEND_URL,
            json={"ussid": st.session_state.ussid, "prompt": prompt},
            stream=True,  # Enable streaming
        ) as response:
            if response.status_code == 200:
                # Stream the response chunk by chunk
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        if decoded_line.startswith("data:"):
                            data = json.loads(decoded_line[5:])
                            # print("😂 data ", data)
                            if "content" in data:
                                full_bot_response += data["content"]
                                # Update the placeholder with the current response
                                bot_response_placeholder.markdown(full_bot_response)
                            elif "error" in data:
                                bot_response_placeholder.markdown(f"**Error:** {data['error']}")
                                break
            else:
                bot_response_placeholder.markdown(f"**Error:** {response.status_code} - {response.text}")
    except Exception as e:
        bot_response_placeholder.markdown(f"**Error:** {str(e)}")

    # Add the full bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_bot_response})