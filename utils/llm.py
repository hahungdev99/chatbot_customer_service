import os
import requests
import ssl 
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from dotenv import load_dotenv
import google.generativeai as genaii

# Load environment variables
load_dotenv()

# Replace with your actual values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GENERATE_MODEL_NAME = os.getenv("GEMINI_GENERATE_MODEL_NAME")


def generate_gemini(system_instruction, user_input):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_GENERATE_MODEL_NAME,
        contents=[
            {
                "role": "user",
                "parts": [{"text": system_instruction}]
            },
            {
                "role": "user", 
                "parts": [{"text": user_input}]
            }
        ]
    )
    return response.text


def generate_gemini_new(system_instruction, user_input):
    # Configure API Key (as shown in Step 2)
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_GENERATE_MODEL_NAME")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genaii.configure(api_key=api_key)

    model = genaii.GenerativeModel(model_name=model_name)
    contents=[
        {
            "role": "user",
            "parts": [{"text": system_instruction}]
        },
        {
            "role": "user", 
            "parts": [{"text": user_input}]
        }
    ]


    try:
        response_complex = model.generate_content(
            contents
            # generation_config=config_high_thinking
        )
        # For code, you might want to inspect parts or check for specific attributes
        return (response_complex.text)
    except Exception as e:
        print(f"An error occurred: {e}")


# def generate_gemini_1(system_instruction, user_input):
#     client = genai.Client(api_key=GEMINI_API_KEY)
#     response = client.models.generate_content(
#         model=GEMINI_GENERATE_MODEL_NAME,
#         contents= user_input,
#         config=GenerateContentConfig(
#         system_instruction=[
#            system_instruction
#         ]),
#     )

#     print(response.text)


# def generate_response(full_chat_history, ussid):
#     try:
#         # Initialize Gemini client
#         client = genai.Client(api_key=GEMINI_API_KEY)

#         # Call Gemini to generate the response with streaming
#         response_stream = client.models.generate_content_stream(
#             model=GEMINI_GENERATE_MODEL_NAME,
#             contents=full_chat_history
#         )

#         collected_chunks = []
#         for chunk in response_stream:
#             if hasattr(chunk, "text"):
#                 content = chunk.text
#                 if content:
#                     collected_chunks.append(content)
#                     yield f"data: {json.dumps({'content': content})}\n\n"

#         # Combine all chunks into the final response
#         full_response = "".join(collected_chunks)

#         # Add the bot's response to the chat history
#         chat_history_store[ussid].append({
#             "role": "assistant",
#             "parts": [{"text": full_response}]
#         })

#     except Exception as e:
#         # Log the exception for debugging
#         traceback.print_exc()
#         yield f"data: {json.dumps({'error': str(e)})}\n\n"

# system_instruction = "you are customer support, your name is HUNG"
# user_input = "what is your name and what is your job "
# # print( generate_gemini(system_instruction, user_input))
# print( generate_gemini_new(system_instruction, user_input))


