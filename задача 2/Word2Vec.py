import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# Загрузка
df_poems = pd.read_csv("processed_poems.csv")
df_songs = pd.read_csv("processed_songs.csv")
df_poems["label"] = 1
df_songs["label"] = 0

df_all = pd.concat([df_songs, df_poems], ignore_index=True)
texts = df_all["text"].fillna("").astype(str)
labels = df_all["label"]

# Токенизация
tokenized_texts = [text.split() for text in texts]

# Обучение Word2Vec
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4, sg=1)
model.save("lyrics_word2vec.model")

# Получение средних векторов для каждого текста
def get_avg_vector(words, model, vector_size):
    vectors = [model.wv[word] for word in words if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

X = np.array([get_avg_vector(words, model, 100) for words in tokenized_texts])
y = labels.values



