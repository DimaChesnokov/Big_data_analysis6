# Big_data_analysis6
Лабораторная работа 6
# Анализ текстов песен и стихов с классификацией

Описание проекта

Проект представляет собой систему анализа текстовых данных (песни и стихи) с применением методов обработки естественного языка (NLP).  
Основные задачи:
- Лемматизация и очистка текста,
- Преобразование текста в векторную форму (Word2Vec),
- Классификация текста как "песня" или "стих" с помощью нескольких моделей (KNN, SVC, RandomForest, LogisticRegression),
- Оценка качества моделей и визуализация смысловых связей между словами.

Проект выполнен в рамках лабораторной работы по дисциплине «Большие данные».

Инструкции по запуску
pip install pandas numpy scikit-learn gensim natasha nltk matplotlib
import nltk
nltk.download('stopwords')
