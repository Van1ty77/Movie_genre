import gradio as gr
import pickle
import re


class ComplexModel:
    def __init__(self):
        self.pipeline = None

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


print("Loading model...")

with open('best_genre_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('genre_encoder.pkl', 'rb') as f:
    genre_encoder = pickle.load(f)

print(f"Model loaded. Genres: {list(genre_encoder.classes_)}")


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_genre(description):
    if not description or description.strip() == "":
        return "Please enter a movie description", None

    cleaned_text = clean_text(description)

    try:
        prediction = model.predict([cleaned_text])
        genre = genre_encoder.inverse_transform(prediction)[0]

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([cleaned_text])[0]
            prob_dict = {
                genre_encoder.classes_[i]: float(probabilities[i])
                for i in range(len(genre_encoder.classes_))
            }
            prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            return genre, prob_dict
        else:
            return genre, None
    except Exception as e:
        return f"Error: {str(e)}", None


examples = [
    "A young boy discovers a magical world and must save it from an evil villain.",
    "Two people fall in love in Paris despite their families trying to keep them apart.",
    "A police detective hunts down a serial killer who leaves cryptic clues at crime scenes.",
    "A group of friends go camping and encounter a terrifying creature in the woods.",
    "A stand-up comedian struggles to make it big in New York City.",
    "The true story of a scientist who changed the world with his invention."
]

custom_css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        background-color: #121212 !important;
    }
    h1 {
        text-align: center;
        color: #f5c518;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 0.2em;
    }
    .imdb-header {
        text-align: center;
        border-bottom: 2px solid #f5c518;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    .imdb-subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 1.3em;
        margin-top: 0;
    }
    .genre-badge {
        background-color: #f5c518;
        color: #000000;
        padding: 6px 16px;
        border-radius: 25px;
        display: inline-block;
        font-weight: 600;
        font-size: 1em;
    }
    .genre-list {
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background-color: #1f1f1f;
        border-radius: 10px;
    }
    .result-box {
        text-align: center !important;
        background-color: #1f1f1f !important;
        padding: 30px !important;
        border-radius: 10px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #f5c518 !important;
    }
    .result-label {
        font-size: 1.2em !important;
        color: #cccccc !important;
        margin-bottom: 10px !important;
    }
    .result-value {
        font-size: 3em !important;
        font-weight: bold !important;
        color: #f5c518 !important;
        letter-spacing: 1px !important;
    }
    footer {
        visibility: hidden;
    }
    label {
        font-size: 1.1em !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    .gr-textbox textarea {
        font-size: 1.1em !important;
        line-height: 1.5 !important;
        background-color: #1f1f1f !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    .gr-textbox textarea:focus {
        border-color: #f5c518 !important;
    }
    .gr-button-primary {
        background-color: #f5c518 !important;
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        padding: 10px 20px !important;
        border: none !important;
    }
    .gr-button-primary:hover {
        background-color: #d4ae14 !important;
    }
    .gr-button-secondary {
        background-color: #333333 !important;
        color: #ffffff !important;
        font-size: 1.1em !important;
        padding: 10px 20px !important;
        border: none !important;
    }
    .gr-button-secondary:hover {
        background-color: #444444 !important;
    }
    h3 {
        color: #ffffff !important;
        font-size: 1.4em !important;
        margin-top: 20px !important;
        margin-bottom: 10px !important;
    }
    .markdown-text {
        color: #cccccc !important;
        font-size: 1em !important;
        background-color: #1f1f1f !important;
        padding: 10px !important;
        border-radius: 8px !important;
        margin: 5px 0 !important;
    }
    .example-block {
        background-color: #1f1f1f !important;
        padding: 12px !important;
        border-radius: 8px !important;
        margin: 8px 0 !important;
        border-left: 3px solid #f5c518 !important;
    }
    .example-text {
        color: #e0e0e0 !important;
        font-size: 0.95em !important;
        margin: 0 !important;
    }
"""

with gr.Blocks(title="IMDb Genre Predictor") as demo:
    gr.HTML("""
        <div class="imdb-header">
            <h1>IMDb</h1>
            <div class="imdb-subtitle">Genre Predictor</div>
        </div>
    """)

    gr.Markdown("""
    <div class="genre-list">
        <span class="genre-badge">Action</span> 
        <span class="genre-badge">Adventure</span> 
        <span class="genre-badge">Animation</span> 
        <span class="genre-badge">Biography</span> 
        <span class="genre-badge">Comedy</span> 
        <span class="genre-badge">Crime</span> 
        <span class="genre-badge">Drama</span> 
        <span class="genre-badge">Horror</span> 
        <span class="genre-badge">Thriller</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Movie Description",
                placeholder="Enter the movie plot or description in English...",
                lines=8
            )

            with gr.Row():
                predict_btn = gr.Button("Predict Genre", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            with gr.Column(elem_classes="result-box"):
                gr.HTML('<div class="result-label">PREDICTED GENRE</div>')
                output_text = gr.Markdown(
                    value="—",
                    elem_classes="result-value"
                )

    with gr.Accordion("Other Possible Genres", open=False):
        output_probabilities = gr.Label(label="Probabilities", num_top_classes=9)

    gr.Markdown("### Example Descriptions")

    for example in examples:
        gr.HTML(f'''
        <div class="example-block">
            <div class="example-text">"{example}"</div>
        </div>
        ''')


    def predict_with_probs(description):
        genre, probs = predict_genre(description)
        return genre, probs


    predict_btn.click(
        fn=predict_with_probs,
        inputs=text_input,
        outputs=[output_text, output_probabilities]
    )

    clear_btn.click(
        fn=lambda: ("", "—", None),
        inputs=[],
        outputs=[text_input, output_text, output_probabilities]
    )

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("SERVER STARTED")
    print("=" * 50)
    print("Local URL: http://127.0.0.1:7860")
    print("=" * 50 + "\n")

    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, css=custom_css, quiet=False)