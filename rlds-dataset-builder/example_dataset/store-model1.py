import tensorflow_hub as hub
import tensorflow as tf

# 下载并加载 Universal Sentence Encoder 模型
model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(model_url)

# 使用模型进行嵌入
sentences = ["This is a test sentence.", "Here is another one."]
embeddings = embed(sentences)

print(embeddings)

model_path = "~/model_tmp/google-usel-5"
tf.saved_model.save(embed, model_path)
print(f"Model saved to {model_path}")