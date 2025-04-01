import pandas as pd
from gensim.models import Word2Vec

# 9. Загрузка предобработанных текстов
df = pd.read_csv("processed_songs.csv")

# Подготовка токенизированных данных
tokenized_texts = [text.split() for text in df["processed_text"]]

# 10. Обучение модели Word2Vec
model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,   # размерность векторов слов
    window=5,          # окно контекста
    min_count=2,       # минимальное число вхождений слова
    workers=4,         # количество потоков
    sg=1               # 1 — skip-gram, 0 — CBOW
)

# Сохраняем модель (опционально)
model.save("song_word2vec.model")

# 11. Проверка: близкие слова к заданному
try:
    similar_words = model.wv.most_similar("порядковый", topn=10)
    print("Ближайшие слова к 'сердце':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.3f}")
except KeyError:
    print("Слово 'сердце' не найдено в словаре модели. Попробуй другое.")
