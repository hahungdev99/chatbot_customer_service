from flask import Flask, request, Response, stream_with_context, jsonify
import json
from dotenv import load_dotenv
import os
import uuid  # For generating unique session IDs
from openai import AzureOpenAI
import tiktoken
from utils.util import *
# from utils.search_local import search_similar_products
from utils.embedding_openai import search_similar_products
import sys

# Load environment variables from .env file
load_dotenv()

# Replace these with your actual values
GPT4_API_KEY = os.getenv("GPT4_API_KEY")
GPT4_ENDPOINT = os.getenv("GPT4_ENDPOINT")
GPT4_DEPLOYMENT_NAME = os.getenv("GPT4_DEPLOYMENT_NAME")
GPT4_API_VERSION = os.getenv("GPT4_API_VERSION")

app = Flask(__name__)



# In-memory storage for chat history (key: ussid, value: list of messages)
chat_history_store = {}

def classify_intent(query):
    """
    Uses GPT-4 to classify the intent of the user's query in Vietnamese.
    
    :param query: The latest query from the user (string).
    :return: 0 for casual chat, 1 for product inquiry.
    """
    # Define the system message for intent classification in Vietnamese
    system_message = f"""
        Bạn là một chuyên gia phân tích ý định khách hàng cho một cửa hàng bán lẻ giày dép và phụ kiện (giày da, giày chạy bộ, dép, thắt lưng,... cho cả nam và nữ). Nhiệm vụ của bạn là **phân loại** câu hỏi của khách hàng thành một trong năm loại sau đây, dựa trên ý định chính của họ:

        **Phân loại ý định:**

        * **'1'**: **Hỏi chuyện thông thường (Chit-chat):** Các câu hỏi mang tính chào hỏi, nói chuyện phiếm, hoặc không liên quan trực tiếp đến sản phẩm, mua hàng, hoặc đơn hàng cụ thể. Ví dụ: "Chào bạn.", "Hôm nay thời tiết thế nào?", "Bạn khỏe không?".

        * **'2'**: **Tìm kiếm sản phẩm theo tên:** Các câu hỏi mà khách hàng đang tìm kiếm thông tin hoặc sự tồn tại của một sản phẩm cụ thể dựa trên tên hoặc mô tả sơ bộ. Ví dụ: "Shop có bán giày sandal nam hai quai chéo không?", "Tôi muốn tìm giày da lười màu nâu.", "Cửa hàng có những loại giày chạy bộ nào?".

        * **'3'**: **Hỏi thông tin chung (vận chuyển, bảo hành,...):** Các câu hỏi liên quan đến các chính sách, quy trình chung của cửa hàng, không cụ thể về một sản phẩm hoặc đơn hàng nào. Ví dụ: "Thời gian giao hàng mất bao lâu?", "Chính sách bảo hành của cửa hàng như thế nào?", "Có những phương thức thanh toán nào?", "chân tôi dài 25cm thì mang size nào ?".

        * **'4'**: **Mong muốn chốt đơn (Mua hàng):** Các câu hỏi hoặc tuyên bố thể hiện ý định mua hàng, đặt hàng, hoặc các hành động liên quan đến việc hoàn tất giao dịch mua sắm. Ví dụ: "Tôi muốn mua đôi giày này.", "Làm thế nào để đặt mua?", "Tôi muốn thanh toán đơn hàng.", "Cho tôi thêm vào giỏ hàng sản phẩm ABC.".

        * **'5'**: **Truy xuất đơn hàng theo số điện thoại:** Các câu hỏi mà khách hàng muốn kiểm tra trạng thái hoặc thông tin chi tiết về đơn hàng đã đặt bằng cách cung cấp số điện thoại. Ví dụ: "Tôi muốn kiểm tra đơn hàng với số điện thoại 09xxxxxxx.", "Đơn hàng của tôi đã giao chưa?", "Thông tin chi tiết về đơn hàng này là gì?".

        **Yêu cầu:**

        Phân tích câu hỏi của khách hàng và trả về **duy nhất một số** tương ứng với loại ý định ( '1', '2', '3', '4', hoặc '5' ). **Không cung cấp bất kỳ giải thích hoặc văn bản nào khác ngoài số phân loại.**

        **Ví dụ:**

        * **Khách hàng:** "Chào shop buổi chiều!"
            **Bạn (LLM):** 1

        * **Khách hàng:** "Tôi đang tìm một đôi giày thể thao màu trắng."
            **Bạn (LLM):** 2

        * **Khách hàng:** "Shop giao hàng đến Hà Nội mất bao lâu?"
            **Bạn (LLM):** 3

        * **Khách hàng:** "Tôi muốn đặt mua ngay đôi giày da mã XYZ size 40."
            **Bạn (LLM):** 4

        * **Khách hàng:** "Cho tôi hỏi về đơn hàng với số điện thoại 03xxxxxxx."
            **Bạn (LLM):** 5

    """

    # Prepare the messages for GPT-4
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    try:
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=GPT4_ENDPOINT,
            api_key=GPT4_API_KEY,
            api_version=GPT4_API_VERSION,
        )

        # Call GPT-4 to classify the intent
        response = client.chat.completions.create(
            model=GPT4_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=1,  # Only need a single token for the classification result
            temperature=0.0,  # Use deterministic output
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )

        # Extract the classification result
        classification_result = response.choices[0].message.content.strip()

        # Convert the result to an integer (0 or 1)
        if classification_result in ["1", "2", "3", "4", "5"]:
            return int(classification_result)
        else:
            # Default to casual chat if the result is unexpected
            print(f" 😢 Result wrong: {classification_result}")
            return 1

    except Exception as e:
        print(f"😒 Error when classification intent of user : {str(e)}")
        return 1  # Default to casual chat in case of errors



@app.route("/chat", methods=["POST"])
def chat():
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
    chat_history_store[ussid].append({"role": "user", "content": user_input})

    # Extract the latest query
    latest_query = chat_history_store[ussid][-1]["content"]

    # Classify the intent of the latest query using GPT-4
    intent = classify_intent(latest_query)
    print(" 😂 intent user ", intent)
    # print("chat_history_store ", chat_history_store)

    # Keep only the latest 10 messages (or 50, depending on your preference)
    chat_history_store[ussid] = chat_history_store[ussid][-10:]


    # chitchat
    if intent==1 or intent==3:
        # Prepare the full chat history to send to the LLM
        system_message = f"""
            Bạn là chatbot tự động có tên là Hunggg, một nhân viên chăm sóc khách hàng thân thiện và hòa đồng tại cửa hàng Giày BQ. Khi khách hàng đưa ra những câu hỏi mang tính chào hỏi, nói chuyện phiếm, hoặc không liên quan trực tiếp đến sản phẩm da (giày, dép, thắt lưng, túi xách), mua hàng, hoặc đơn hàng cụ thể, nhiệm vụ của bạn là phản hồi một cách lịch sự, duy trì một cuộc trò chuyện ngắn gọn và thân thiện.

            **Hướng dẫn:**

            1.  Đọc kỹ câu hỏi của khách hàng.
            2.  Nếu câu hỏi thuộc loại chào hỏi hoặc nói chuyện phiếm (ví dụ: "Chào bạn.", "Hôm nay thời tiết thế nào?", "Bạn khỏe không?"), hãy đáp lại một cách lịch sự và ngắn gọn. Ví dụ: "Chào quý khách!", "Vâng, thời tiết hôm nay khá đẹp ạ.", "Tôi khỏe, cảm ơn quý khách!".
            3.  Bạn có thể hỏi một câu hỏi mở để tiếp tục cuộc trò chuyện ở mức độ xã giao nếu phù hợp, nhưng tránh đi sâu vào các chủ đề cá nhân hoặc không liên quan đến cửa hàng. Ví dụ: "Quý khách có dự định gì cho ngày hôm nay không ạ?" (nếu ngữ cảnh cho phép).
            4.  Hãy chú ý lắng nghe nếu khách hàng có ý định chuyển sang hỏi về sản phẩm hoặc dịch vụ sau phần chào hỏi.
            5.  Nếu cuộc trò chuyện có vẻ không có mục đích cụ thể và kéo dài, bạn có thể kết thúc một cách lịch sự bằng cách hỏi: "Tôi có thể hỗ trợ gì thêm cho quý khách không ạ?" hoặc "Nếu quý khách có bất kỳ câu hỏi nào khác về sản phẩm, đừng ngần ngại cho chúng tôi biết nhé."

            """
        
    # products
    elif intent == 2:
        result_search = search_similar_products(latest_query, top_k=5)
        print("😂 result search ", result_search)
        system_message = f"""
            Bạn là chatbot tự động có tên là Hunggg. Bạn là một nhân viên chăm sóc khách hàng chuyên nghiệp và am hiểu về các sản phẩm giày dép và phụ kiện của cửa hàng Giày BQ. Bạn sẽ sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của khách hàng một cách chính xác và hữu ích.
            **Câu hỏi của khách hàng:** 
            {latest_query}
            **Thông tin sản phẩm liên quan:** 
            {result_search}
            **Hướng dẫn:**

            1.  Đọc kỹ câu hỏi của khách hàng để hiểu rõ nhu cầu của họ.
            2.  Xem xét các thông tin về sản phẩm liên quan được cung cấp. Các sản phẩm này có thể tương tự hoặc liên quan đến sản phẩm mà khách hàng đề cập.
            3.  Sử dụng thông tin từ các sản phẩm liên quan để trả lời câu hỏi của khách hàng một cách chi tiết và hữu ích nhất có thể.
            4.  Nếu câu hỏi của khách hàng liên quan đến một sản phẩm cụ thể (ví dụ: "AD147"), hãy ưu tiên cung cấp thông tin về các sản phẩm tương tự được tìm thấy.
            5.  Bạn có thể cung cấp thông tin về:
                * Tên sản phẩm
                * Mô tả ngắn gọn
                * Chất liệu
                * Kiểu dáng
                * Màu sắc
                * Kích cỡ hiện có
                * Giá cả
                * Các tính năng nổi bật
                * So sánh với sản phẩm mà khách hàng đã hỏi (nếu phù hợp)
                * Gợi ý các sản phẩm phù hợp khác dựa trên sở thích hoặc nhu cầu tiềm năng của khách hàng.
            6.  Nếu không có sản phẩm liên quan nào phù hợp hoặc thông tin không đủ để trả lời, hãy trả lời một cách lịch sự rằng bạn cần thêm thông tin hoặc sẽ kiểm tra lại. Tránh đưa ra thông tin không chính xác.
            7.  Luôn giữ thái độ lịch sự và chuyên nghiệp trong suốt cuộc trò chuyện.
            """
    
    elif intent == 30:
        result_search = search_similar_products(latest_query, top_k=5)
        system_message = f"""
            Bạn là chatbot tự động có tên là Hunggg. Bạn là một nhân viên chăm sóc khách hàng chuyên nghiệp tại cửa hàng Giày BQ. Nhiệm vụ của bạn là giải đáp các thắc mắc chung của khách hàng liên quan đến chính sách và quy trình của cửa hàng.

            **Thông tin về chính sách và quy trình của cửa hàng:**
            {result_search}

            **Câu hỏi của khách hàng:**
            {latest_query}

            **Hướng dẫn:**

            1.  Đọc kỹ câu hỏi của khách hàng để xác định loại thông tin chung mà họ đang tìm kiếm (ví dụ: vận chuyển, bảo hành, thanh toán, đổi trả, hướng dẫn chọn size).
            2.  Tham khảo phần "Thông tin về chính sách và quy trình của cửa hàng" (`Thông tin về chính sách và quy trình của cửa hàng`) để tìm kiếm câu trả lời phù hợp.
            3.  Cung cấp câu trả lời chi tiết, rõ ràng và dễ hiểu cho khách hàng dựa trên thông tin đã được cung cấp.
            4.  Nếu câu hỏi liên quan đến một chủ đề cụ thể, hãy tập trung vào thông tin liên quan đến chủ đề đó.
            5.  Đối với các câu hỏi về kích cỡ (ví dụ: "chân tôi dài 25cm thì mang size nào?"), hãy tham khảo bảng quy đổi kích cỡ giày của cửa hàng (nếu có trong `Thông tin về chính sách và quy trình của cửa hàng`) và đưa ra gợi ý phù hợp.
            6.  Nếu thông tin trong `Thông tin về chính sách và quy trình của cửa hàng` không đầy đủ hoặc không có câu trả lời chính xác cho câu hỏi của khách hàng, hãy trả lời một cách lịch sự rằng bạn cần kiểm tra lại hoặc sẽ cung cấp thông tin sau. Tránh đưa ra thông tin không chắc chắn.
            7.  Luôn giữ thái độ lịch sự và sẵn sàng hỗ trợ thêm nếu khách hàng có thêm thắc mắc.

        """

    # order
    elif intent == 4:
        system_message = f"""
            Bạn là chatbot tự động có tên là Hunggg. Bạn là một nhân viên chăm sóc khách hàng chuyên nghiệp và am hiểu về các sản phẩm giày dép và phụ kiện của cửa hàng. Bạn có nhiệm vụ hỗ trợ tổng hợp thông tin từ lịch sử trò chuyện với khách hàng để chuẩn bị cho việc tạo đơn hàng. Hãy đọc kỹ lịch sử trò chuyện và trích xuất các thông tin sau nếu có:

            **Thông tin cần trích xuất:**

            * **Tên khách hàng:** Tìm kiếm bất kỳ thông tin nào đề cập đến tên của khách hàng.
            * **Địa chỉ giao hàng:** Tìm kiếm thông tin về địa chỉ mà khách hàng muốn nhận hàng.
            * **Số điện thoại liên hệ:** Tìm kiếm số điện thoại mà khách hàng đã cung cấp.
            * **Sản phẩm quan tâm/muốn mua:** Xác định rõ tên sản phẩm, mã sản phẩm (nếu có), số lượng, màu sắc, kích cỡ hoặc bất kỳ đặc điểm cụ thể nào khác mà khách hàng muốn mua.
            * **Các yêu cầu đặc biệt khác:** Ghi nhận bất kỳ yêu cầu đặc biệt nào khác của khách hàng liên quan đến đơn hàng (ví dụ: gói quà, ghi chú đặc biệt).

            **Lịch sử trò chuyện:**
                {chat_history_store[ussid]}
            
            **Hướng dẫn:**
            1.  Đọc kỹ toàn bộ lịch sử trò chuyện được cung cấp.
            2.  Tìm kiếm và trích xuất các thông tin được liệt kê ở trên.
            3.  Nếu một thông tin xuất hiện nhiều lần hoặc được cập nhật, hãy ưu tiên thông tin mới nhất.
            4.  Nếu một thông tin không có trong lịch sử trò chuyện, hãy bỏ qua trường đó.
            5.  Trình bày thông tin đã trích xuất một cách rõ ràng và có cấu trúc.
            6.  Gửi thông tin kèm url="https://giaybq.com.vn/cart" để khách hàng có thể điền thông tin thanh toán

            **Định dạng đầu ra mong muốn:**
            ```json
            {{
            "tên_khách_hàng": "{{tên khách hàng trích xuất được}}",
            "địa_chỉ_giao_hàng": "{{địa chỉ giao hàng trích xuất được}}",
            "số_điện_thoại": "{{số điện thoại trích xuất được}}",
            "sản_phẩm_muốn_mua": [
                {{
                "tên_sản_phẩm": "{{tên sản phẩm 1}}",
                "mã_sản_phẩm": "{{mã sản phẩm 1 (nếu có)}}",
                "số_lượng": "{{số lượng sản phẩm 1}}",
                "màu_sắc": "{{màu sắc sản phẩm 1 (nếu có)}}",
                "kích_cỡ": "{{kích cỡ sản phẩm 1 (nếu có)}}",
                "ghi_chú": "{{ghi chú khác về sản phẩm 1 (nếu có)}}"
                }},
                {{
                "tên_sản_phẩm": "{{tên sản phẩm 2}}",
                "mã_sản_phẩm": "{{mã sản phẩm 2 (nếu có)}}",
                "số_lượng": "{{số lượng sản phẩm 2}}",
                "màu_sắc": "{{màu sắc sản phẩm 2 (nếu có)}}",
                "kích_cỡ": "{{kích cỡ sản phẩm 2 (nếu có)}}",
                "ghi_chú": "{{ghi chú khác về sản phẩm 2 (nếu có)}}"
                }}
            ],
            "yêu_cầu_đặc_biệt": "{{các yêu cầu đặc biệt khác (nếu có)}}"
            }}
        """

    elif intent==5:
        system_message = f"""
            Bạn là chatbot tự động có tên là Hunggg. Bạn là một nhân viên chăm sóc khách hàng chuyên nghiệp và am hiểu về các sản phẩm giày dép và phụ kiện của cửa hàng Giày BQ. Bạn sẽ sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của khách hàng một cách chính xác và hữu ích.

            **Câu hỏi của khách hàng:**     
            {latest_query}

            1.  Đọc kỹ câu hỏi của khách hàng để xác định rằng họ đang muốn kiểm tra thông tin đơn hàng và có thể cung cấp số điện thoại hoặc đề cập đến việc kiểm tra đơn hàng.
            2.  Chào hỏi khách hàng một cách lịch sự và thân thiện.
            3.  Thông báo cho khách hàng rằng họ có thể dễ dàng kiểm tra trạng thái đơn hàng của mình thông qua liên kết trên trang web của cửa hàng.
            4.  Cung cấp đường dẫn (URL) chính xác đến trang kiểm tra đơn hàng.
            5.  Khuyến khích khách hàng sử dụng liên kết này để xem thông tin chi tiết về đơn hàng của họ.
            6.  Kết thúc bằng một lời chúc hoặc một câu hỏi gợi ý nếu họ cần thêm sự hỗ trợ sau khi kiểm tra.

            **Thông tin liên kết kiểm tra đơn hàng:**

            Trang web kiểm tra đơn hàng: `https://giaybq.com.vn/pages/tra-cuu-don-hang-online`

            **Ví dụ:**

            * **Khách hàng:** "Tôi muốn kiểm tra đơn hàng của tôi với số điện thoại 090xxxxxxx."
                **Bạn (LLM):** "Chào quý khách! Để kiểm tra thông tin chi tiết về đơn hàng của mình, quý khách vui lòng truy cập đường dẫn sau: hung.com/kiem-tra-don-hang. Quý khách có thể theo dõi trạng thái giao hàng và xem các thông tin liên quan tại đó. Nếu quý khách cần hỗ trợ thêm sau khi kiểm tra, đừng ngần ngại cho chúng tôi biết nhé."

            * **Khách hàng:** "Đơn hàng của tôi đã giao chưa?"
                **Bạn (LLM):** "Chào quý khách! Quý khách có thể kiểm tra trạng thái giao hàng của đơn hàng một cách nhanh chóng tại trang web của chúng tôi: hung.com/kiem-tra-don-hang. Xin quý khách vui lòng truy cập để xem thông tin cập nhật nhất. Nếu cần thêm thông tin, xin vui lòng cho chúng tôi biết số điện thoại hoặc mã đơn hàng để được hỗ trợ tốt hơn."
        """


    full_chat_history = [{"role": "system", "content": system_message}] + chat_history_store[ussid]
    # print("😒 full_chat_history ", full_chat_history)

    try:
        # Calculate token count for the input messages
        input_token_count = calculate_token_count(full_chat_history)

        def generate_response():
            nonlocal input_token_count
            try:
                # Initialize Azure OpenAI client
                client = AzureOpenAI(
                    azure_endpoint=GPT4_ENDPOINT,
                    api_key=GPT4_API_KEY,
                    api_version=GPT4_API_VERSION,
                )

                # Call GPT-4 to generate the response
                response = client.chat.completions.create(
                    model=GPT4_DEPLOYMENT_NAME,
                    messages=full_chat_history,
                    max_tokens=2000,
                    temperature=0.7,
                    top_p=0.99,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=True  # Enable streaming
                )

                collected_chunks = []
                output_token_count = 0
                for chunk in response:
                    if chunk.choices and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                        content = chunk.choices[0].delta.content
                        if content:
                            collected_chunks.append(content)
                            yield f"data: {json.dumps({'content': content})}\n\n"

                            # Incrementally calculate output token count
                            output_token_count += len(tiktoken.encoding_for_model("gpt-4").encode(content))

                # Combine all chunks into the final response
                full_response = "".join(collected_chunks)

                # Add the bot's response to the chat history
                chat_history_store[ussid].append({"role": "assistant", "content": full_response})

                # Log total token usage
                print(f"Input Tokens: {input_token_count}, Output Tokens: {output_token_count}")

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate_response()), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)