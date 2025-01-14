import tensorflow_hub as hub

# 下载并加载 Universal Sentence Encoder 模型
model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(model_url)

# 使用模型进行嵌入
sentences = ["This is a test sentence.", "Here is another one."]
embeddings = embed(sentences)

print(embeddings)

model_path = "/mnt/data-qilin/ovla/dataset/0114-TestRLDS/google-universal-sentence-encoder-large-5"
tf.saved_model.save(embed, model_path)