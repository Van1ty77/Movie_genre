import pytest
import os
import re
import joblib
import pickle


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_model():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'genre_pipeline.joblib')
    encoder_path = os.path.join(project_dir, 'genre_encoder.pkl')

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)

    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    return model, encoder


def test_clean_text():
    result = clean_text("Hello, World!!! This is a TEST.")
    expected = "hello world this is a test"
    assert result == expected


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == "none"


def test_clean_text_numbers():
    result = clean_text("Movie 2: The Sequel 2024")
    expected = "movie the sequel"
    assert result == expected


def test_model_files_exist():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_dir, 'genre_pipeline.joblib')
    encoder_path = os.path.join(project_dir, 'genre_encoder.pkl')

    assert os.path.exists(model_path), f"Model not found: {model_path}"
    assert os.path.exists(encoder_path), f"Encoder not found: {encoder_path}"


def test_model_prediction():
    model, encoder = load_model()

    if model is None or encoder is None:
        pytest.skip("Model files not found. Run train.py first.")

    description = "A young boy discovers a magical world and must save it from an evil villain."
    cleaned = clean_text(description)

    prediction = model.predict([cleaned])
    genre = encoder.inverse_transform(prediction)[0]

    assert isinstance(genre, str)
    assert len(genre) > 0


def test_model_returns_valid_genre():
    model, encoder = load_model()

    if model is None or encoder is None:
        pytest.skip("Model files not found. Run train.py first.")

    test_descriptions = [
        "Two people fall in love in Paris.",
        "A police detective hunts a killer.",
        "A group of friends encounter a monster.",
        "A stand-up comedian in New York.",
    ]

    valid_genres = list(encoder.classes_)

    for desc in test_descriptions:
        cleaned = clean_text(desc)
        prediction = model.predict([cleaned])
        genre = encoder.inverse_transform(prediction)[0]

        assert genre in valid_genres, f"Genre '{genre}' not in {valid_genres}"


def test_probabilities():
    model, encoder = load_model()

    if model is None or encoder is None:
        pytest.skip("Model files not found. Run train.py first.")

    description = "A police detective hunts a killer."
    cleaned = clean_text(description)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([cleaned])[0]
        assert len(probs) == len(encoder.classes_)
        assert abs(sum(probs) - 1.0) < 0.01


def test_all_genres_available():
    model, encoder = load_model()

    if model is None or encoder is None:
        pytest.skip("Model files not found. Run train.py first.")

    expected_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Drama', 'Horror',
                       'Thriller']

    for genre in expected_genres:
        assert genre in encoder.classes_, f"Genre '{genre}' not found in encoder"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])