import pandas as pd
import numpy as np
import json
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from collections import Counter
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("\nЗагрузка данных...")
df = pd.read_csv("IMDb movies.csv")
print(f"Размер датасета: {df.shape}")

print("\nОпределение признаков и целевой переменной:")
print("X (признаки) -> description (текстовое описание фильма)")
print("y (целевая переменная) -> genre (жанр фильма)")

df['main_genre'] = df['genre'].str.split(',').str[0].str.strip()
df = df[df['main_genre'].notna()]
df = df[df['main_genre'] != '']

top_genres = df['main_genre'].value_counts().head(9).index
df = df[df['main_genre'].isin(top_genres)]
print(f"Выбрано жанров: {len(top_genres)}")

print("\nПреобразование текста в числа (TF-IDF)...")

df['description'] = df['description'].fillna('').astype(str)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['description'] = df['description'].apply(clean_text)
df = df[df['description'].str.len() > 20]
print(f"После очистки осталось: {len(df)} фильмов")

print("\nМасштабирование признаков:")
print("TF-IDF автоматически нормализует текстовые признаки")

y_encoder = LabelEncoder()
y = df['main_genre']
y_encoded = y_encoder.fit_transform(y)
print(f"Жанры закодированы числами: {list(enumerate(y_encoder.classes_))}")

print("\nРазделение данных на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(
    df['description'], y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Обучающая выборка: {len(X_train)} примеров (80%)")
print(f"Тестовая выборка (сейф): {len(X_test)} примеров (20%)")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True
)

print("\nВыбраны два принципиально разных алгоритма:")
print("1) LogisticRegression - сложный алгоритм из библиотеки scikit-learn")
print("2) SimpleKeywordClassifier - собственное простое правило")


class ComplexModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


class SimpleKeywordClassifier:
    def __init__(self):
        self.genre_keywords = {
            'Action': ['fight', 'explosion', 'chase', 'battle', 'war', 'gun', 'shoot', 'rescue', 'attack', 'escape'],
            'Comedy': ['funny', 'joke', 'laugh', 'humor', 'hilarious', 'comedy', 'fun', 'crazy', 'silly', 'wit'],
            'Drama': ['emotional', 'family', 'love', 'relationship', 'struggle', 'conflict', 'loss', 'pain', 'hope',
                      'tear'],
            'Horror': ['scary', 'horror', 'terror', 'monster', 'ghost', 'zombie', 'kill', 'death', 'fear', 'nightmare'],
            'Crime': ['crime', 'murder', 'detective', 'police', 'criminal', 'investigation', 'robbery', 'gang',
                      'mystery', 'kill'],
            'Adventure': ['adventure', 'journey', 'quest', 'explore', 'expedition', 'treasure', 'discover', 'travel',
                          'wild', 'island'],
            'Animation': ['animated', 'cartoon', 'animation', 'cgi', 'pixar', 'disney', 'draw', 'character', 'kid',
                          'colorful'],
            'Biography': ['biography', 'real', 'true', 'based', 'life', 'historical', 'story of', 'document', 'born',
                          'died'],
            'Thriller': ['thriller', 'suspense', 'twist', 'mystery', 'psychological', 'dark', 'tense', 'unexpected',
                         'secret', 'hidden']
        }
        self.genre_list = list(self.genre_keywords.keys())
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.genre_list)

    def fit(self, X, y):
        return self

    def predict(self, X):
        predictions = []
        for text in X:
            text_lower = str(text).lower()
            scores = {}
            for genre, keywords in self.genre_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                scores[genre] = score
            best_genre = max(scores, key=scores.get)
            predictions.append(best_genre)
        return self.label_encoder.transform(predictions)


print("\nЗапуск кросс-валидации для обеих моделей...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': ComplexModel(),
    'SimpleKeywordClassifier': SimpleKeywordClassifier()
}

cv_results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    print(f"Кросс-валидация (5 фолдов)...")

    cv_scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        if 'Logistic' in name:
            model_clone = ComplexModel()
        else:
            model_clone = SimpleKeywordClassifier()

        model_clone.fit(X_tr, y_tr)
        pred = model_clone.predict(X_val)
        cv_scores.append(accuracy_score(y_val, pred))

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_test, y_pred_val)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {val_acc:.4f}")

    print("\nМетрики по классам (на тестовой выборке):")
    for i, genre in enumerate(y_encoder.classes_):
        tp = np.sum((y_test == i) & (y_pred_val == i))
        fp = np.sum((y_test != i) & (y_pred_val == i))
        fn = np.sum((y_test == i) & (y_pred_val != i))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0

        print(f"{genre:12} - Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}")

    cv_results[name] = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_acc': train_acc,
        'test_acc': val_acc,
        'model': model,
        'predictions': y_pred_val
    }

print("\nВизуализация ошибок...")

best_name = max(cv_results, key=lambda x: cv_results[x]['test_acc'])
best_result = cv_results[best_name]
print(f"\nЛучшая модель по тестовой точности: {best_name}")

y_pred_best = best_result['predictions']
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=y_encoder.classes_,
            yticklabels=y_encoder.classes_)
plt.title(f'Матрица ошибок - {best_name}')
plt.xlabel('Предсказанный жанр')
plt.ylabel('Истинный жанр')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

error_pairs = []
for i in range(len(y_test)):
    if y_test[i] != y_pred_best[i]:
        true_genre = y_encoder.inverse_transform([y_test[i]])[0]
        pred_genre = y_encoder.inverse_transform([y_pred_best[i]])[0]
        error_pairs.append((true_genre, pred_genre))

error_counts = Counter(error_pairs)
print("\nТоп-10 самых частых ошибок (истинный → предсказанный):")
for (true, pred), count in error_counts.most_common(10):
    print(f"{true:15} → {pred:15}: {count:4} раз")

print("\nВыбор лучшей модели...")
for name, res in cv_results.items():
    print(f"{name}: CV={res['cv_mean']:.4f}±{res['cv_std']:.4f}, Test={res['test_acc']:.4f}")

final_model_name = max(cv_results, key=lambda x: cv_results[x]['cv_mean'])
final_model = cv_results[final_model_name]['model']
print(f"\nВыбрана модель: {final_model_name}")

print("\nПроверка на отложенных данных (сейф)...")
final_predictions = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)
final_f1_macro = f1_score(y_test, final_predictions, average='macro')
final_f1_weighted = f1_score(y_test, final_predictions, average='weighted')

print(f"Точность на сейфе: {final_accuracy:.4f}")
print(f"F1-score (macro): {final_f1_macro:.4f}")
print(f"F1-score (weighted): {final_f1_weighted:.4f}")

print("\nСохранение модели и параметров...")

with open('best_genre_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Модель сохранена в 'best_genre_model.pkl'")

with open('genre_encoder.pkl', 'wb') as f:
    pickle.dump(y_encoder, f)
print("Кодировщик жанров сохранён в 'genre_encoder.pkl'")

model_params = {
    'model_name': final_model_name,
    'model_type': 'LogisticRegression with TF-IDF' if 'Logistic' in final_model_name else 'SimpleKeywordClassifier',
    'test_accuracy': float(final_accuracy),
    'test_f1_macro': float(final_f1_macro),
    'test_f1_weighted': float(final_f1_weighted),
    'num_genres': len(y_encoder.classes_),
    'genres': list(y_encoder.classes_),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'cv_results': {
        name: {
            'mean': float(res['cv_mean']),
            'std': float(res['cv_std']),
            'test_acc': float(res['test_acc'])
        } for name, res in cv_results.items()
    },
    'most_common_errors': [
        {'true': true, 'pred': pred, 'count': count}
        for (true, pred), count in error_counts.most_common(5)
    ]
}

with open('model_parameters.json', 'w', encoding='utf-8') as f:
    json.dump(model_params, f, indent=4, ensure_ascii=False)
print("Параметры сохранены в 'model_parameters.json'")

print(f"\nЛучшая модель: {final_model_name}")
print(f"Ключевая метрика (Test Accuracy) на новых данных: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%)")
print(f"Дополнительная метрика (F1-macro): {final_f1_macro:.4f}")

if error_counts:
    most_common = error_counts.most_common(1)[0]
    true_class, pred_class = most_common[0]
    print(f"Чаще всего модель путает: '{true_class}' и '{pred_class}' ({most_common[1]} ошибок)")

print(f"\nСохранённые файлы:")
print("1. best_genre_model.pkl - обученная модель")
print("2. genre_encoder.pkl - кодировщик жанров")
print("3. model_parameters.json - параметры и метрики модели")