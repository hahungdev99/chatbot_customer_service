import os
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

# 1. Khai báo tên model và đường dẫn lưu trữ local
model_name = 'dangvantuan/vietnamese-embedding'
save_path = os.path.join('llm_models', model_name) # Tạo đường dẫn: models/vietnamese-embedding

# 2. Kiểm tra xem model đã được lưu local chưa
if not os.path.exists(save_path):
    print(f"Model chưa có tại '{save_path}'")
    print("Tiến hành tải model từ Hugging Face...")
    # Nếu chưa có, tải model từ Hugging Face
    model = SentenceTransformer(model_name)
    # Và lưu lại vào đường dẫn đã khai báo
    model.save(save_path)
    print(f"Đã lưu model vào '{save_path}'")
else:
    print(f"Model đã có sẵn. Tải model từ '{save_path}'")
    # Nếu đã có, chỉ cần tải model từ thư mục local
    model = SentenceTransformer(save_path)

# --- Phần code của bạn giữ nguyên ---
sentences = ["Hà Nội là thủ đô của Việt Nam", "Đà Nẵng là thành phố du lịch"]
tokenizer_sent = [ViTokenizer.tokenize(sent) for sent in sentences]

# 3. Sử dụng model đã được tải
embeddings = model.encode(tokenizer_sent)
print("\nKết quả embedding:")
print(embeddings)