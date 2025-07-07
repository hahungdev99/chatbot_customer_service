from sentence_transformers import SentenceTransformer
sentences = ["Cô giáo đang ăn kem", "Chị gái đang thử món thịt dê"]

model = SentenceTransformer('bkai-foundation/vietnamese-nb-sbert')
embeddings = model.encode(sentences)
print(embeddings)
