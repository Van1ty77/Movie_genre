# IMDb Genre Predictor

Классификация жанров фильмов по текстовому описанию.

## Установка и запуск

1. Клонируйте репозиторий
2. Создайте виртуальное окружение:
python -m venv .venv
.venv\Scripts\activate

3. Установите зависимости:
pip install -r requirements.txt

4. Обучите модель (если нет файлов модели):
python train.py

5. Запустите веб-интерфейс:
python app.py

6. Откройте в браузере: http://127.0.0.1:7860

## Тестирование

Запуск тестов:
pytest tests/test_basic.py -v

## Технологии

- Python 3.10+
- Gradio — веб-интерфейс
- scikit-learn — обучение модели (TF-IDF + LogisticRegression)
- Pandas, NumPy — обработка данных