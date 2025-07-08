from flask import Flask, request, Response, stream_with_context, jsonify
import json
from dotenv import load_dotenv
import os
import traceback
from google import genai
from utils.init_system_prompt import *
from utils.util import classify_intent_gemini, classify_intent_history_gemini
from utils.embedding_vnembedding import search_similar_products_vnembedding, search_similar_products_vnembedding_mongodb
# from utils.embedding_sbert import search_similar_products_sbert
from utils.embedding_gemini import search_similar_products_gemini
from utils.llm import generate_gemini
from utils.embedding_gemini_docs import search_docs_gemini
import pandas as pd

# Configure Pandas display options
pd.set_option('display.max_colwidth', None)  # Show full content of columns
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.width', 1000)         # Set display width for better readability

# Load environment variables from .env file
load_dotenv()

# Replace these with your actual values
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GENERATE_MODEL_NAME = os.getenv("GEMINI_GENERATE_MODEL_NAME")

app = Flask(__name__)

# In-memory storage for chat history (key: ussid, value: list of messages)
chat_history_store = {}


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        ussid = data.get("ussid")  # Unique session ID
        user_input = data.get("prompt", "")

        # Validate ussid
        if not ussid:
            return jsonify({"error": "Session ID (ussid) is required."}), 400

        # Retrieve or initialize chat history for this session
        if ussid not in chat_history_store:
            chat_history_store[ussid] = []

        # Add the user's message to the chat history
        chat_history_store[ussid].append({
            "role": "user",
            "parts": [{"text": user_input}]
        })

        # Extract the latest query
        latest_query = chat_history_store[ussid][-1]["parts"][0]["text"]

        print("😂 latest_query ", latest_query)

        # Classify the intent of the latest query using Gemini
        intent_user = classify_intent_gemini(latest_query)
        print(" 😂 intent user ", intent_user)

        # # Classify the intent of the latest query using Gemini with history chat
        # intent_user = classify_intent_history_gemini(latest_query, chat_history_store[ussid]) 
        # print(" 😂 intent user ", intent_user)


        # Keep only the latest 10 messages (or 50, depending on your preference)
        chat_history_store[ussid] = chat_history_store[ussid][-5:]
 
        # print('👌 chat history ', history_str)

        # Convert chat history to the correct format for Gemini
        full_chat_history = chat_history_store[ussid]

        # print("😂 chat_history_store: ", chat_history_store)

        match intent_user:
            # case 1: chatchit
            case 1: 
                system_prompt = chatchit_prompt()
            
            # case 2: support find information of product
            case 2:
                system_prompt_tmp = get_product_infor_from_query(latest_query)
                product_infor = generate_gemini(system_prompt_tmp, latest_query)
                print("👌 user query -> product infor : ", product_infor)

                result_search = search_similar_products_vnembedding_mongodb(column_name="name", query= product_infor, top_k=5)
                print("👌 result_search ", result_search)
                system_prompt = anwswer_product_infor_prompt(latest_query, result_search)

            # case 3: answer other question relate to shop
            case 3:
                system_prompt_tmp = get_shop_infor_from_query(latest_query)
                shop_infor = generate_gemini(system_prompt_tmp, latest_query)
                print("👌 user query -> shop infor : ", shop_infor)

                result_search = search_docs_gemini(shop_infor)
                system_prompt = answer_shop_infor_prompt(latest_query, result_search)

            # case 4: gather information and support create order
            case 4:
                url_payment = "https://giaybq.com.vn/cart"
                system_prompt = create_order_prompt(chat_history_store, ussid, url_payment)

            # case 5: tracking order depend on phone number    
            case 5:
                url_tracking_order = "https://giaybq.com.vn/pages/tra-cuu-don-hang-online"
                system_prompt = tracking_order_prompt(latest_query, url_tracking_order)

        # Add system message
        full_chat_history.insert(0, {
            "role": "user",
            "parts": [{"text": system_prompt}]
        })

        # print("😂 full_chat_history ", full_chat_history)
        # Return the streaming response
        return Response(stream_with_context(generate_response(full_chat_history, ussid)), mimetype="text/event-stream")

    except Exception as e:
        # Log the exception for debugging
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def generate_response(full_chat_history, ussid):
    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Call Gemini to generate the response with streaming
        response_stream = client.models.generate_content_stream(
            model=GEMINI_GENERATE_MODEL_NAME,
            contents=full_chat_history
        )
        collected_chunks = []
        for chunk in response_stream:
            if hasattr(chunk, "text"):
                content = chunk.text
                if content:
                    collected_chunks.append(content)
                    yield f"data: {json.dumps({'content': content})}\n\n"

        # Combine all chunks into the final response
        full_response = "".join(collected_chunks)

        # Add the bot's response to the chat history
        chat_history_store[ussid].append({
            "role": "assistant",
            "parts": [{"text": full_response}]
        })

    except Exception as e:
        # Log the exception for debugging
        traceback.print_exc()
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)