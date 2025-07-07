import os
import tiktoken
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from utils.init_system_prompt import * 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Replace with your actual values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GENERATE_MODEL_NAME = os.getenv("GEMINI_GENERATE_MODEL_NAME")

# Function to calculate token count
def calculate_token_count(messages, model_name="gpt-4"):
    """
    Calculate the total number of tokens in a list of messages.
    
    :param messages: List of message dictionaries (e.g., [{"role": "user", "content": "Hello"}]).
    :param model_name: The name of the model to use for tokenization (default: "gpt-4").
    :return: Total token count.
    """
    # Load the tokenizer for the specified model
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: Model not found. Using default tokenizer.")
        encoding = tiktoken.get_encoding("cl100k_base")  # Default tokenizer for GPT-4 and GPT-3.5

    total_tokens = 0
    for message in messages:
        # Each message has a role and content
        total_tokens += len(encoding.encode(message["content"]))
        total_tokens += 4  # Additional tokens for metadata (role, etc.)
    return total_tokens



def classify_intent_gemini(query):
    # Define the system message for intent classification in Vietnamese
    system_message = classify_intent_prompt()

    chat_messages = [
        {
            "role": "user",
            "parts": [{"text": system_message}]
        },      
        {
            "role": "user",
            "parts": [{"text": query}]
        }]

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Call Gemini to generate the response with streaming
        response = client.models.generate_content(
            model=GEMINI_GENERATE_MODEL_NAME,
            contents=chat_messages
        )        

        # print(response.text)
        # Extract the classification result
        classification_result = response.text.strip()
        # print("classification_result ", len(classification_result))
        # Convert the result to an integer
        if classification_result in ["1", "2", "3", "4", "5"]:
            return int(classification_result)
        else:
            # Default to casual chat if the result is unexpected
            print(f" 😢 Result wrong: {classification_result}")
            return 1

    except Exception as e:
        print(f"😒 Error when classification intent of user : {str(e)}")
        return 1  # Default to casual chat in case of errors


def classify_intent_history_gemini(latest_query, chat_history):
    # Define the system message for intent classification in Vietnamese
    system_message = classify_intent_prompt_history(latest_query, chat_history)

    chat_messages = [
        {
            "role": "user",
            "parts": [{"text": system_message}]
        },      
        {
            "role": "user",
            "parts": [{"text": latest_query}]
        }]

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Call Gemini to generate the response with streaming
        response = client.models.generate_content(
            model=GEMINI_GENERATE_MODEL_NAME,
            contents=chat_messages
        )        

        # print(response.text)
        # Extract the classification result
        classification_result = response.text.strip()
        # print("classification_result ", len(classification_result))
        # Convert the result to an integer
        if classification_result in ["1", "2", "3", "4", "5"]:
            return int(classification_result)
        else:
            # Default to casual chat if the result is unexpected
            print(f" 😢 Result wrong: {classification_result}")
            return 1

    except Exception as e:
        print(f"😒 Error when classification intent of user : {str(e)}")
        return 1  # Default to casual chat in case of errors




