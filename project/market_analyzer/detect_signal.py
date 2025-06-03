import os
import joblib
import xgboost as xgb
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from datetime import datetime

MODEL_DIR = 'market_analyzer/models'
_selected = {'model': None, 'vectorizer': None}

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.json')]

def list_vectorizers():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]

def select_model_interactively():
    """
    Показывает виджеты для выбора модели и векторизатора.
    Сохраняет выбранные объекты в глобальное хранилище _selected.
    """
    model_files = list_models()
    vectorizer_files = list_vectorizers()

    if not model_files:
        raise FileNotFoundError("Нет доступных моделей .json в папке models/")
    if not vectorizer_files:
        raise FileNotFoundError("Нет доступных векторизаторов .joblib в папке models/")

    model_dropdown = widgets.Dropdown(
        options=model_files,
        description='Выбор модели:'
    )

    vectorizer_dropdown = widgets.Dropdown(
        options=vectorizer_files,
        description='TF-IDF векторизатор:'
    )

    load_button = widgets.Button(description="Загрузить модель")
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()
            model_path = os.path.join(MODEL_DIR, model_dropdown.value)
            vectorizer_path = os.path.join(MODEL_DIR, vectorizer_dropdown.value)

            model = xgb.XGBClassifier()
            model.load_model(model_path)
            vectorizer = joblib.load(vectorizer_path)

            _selected['model'] = model
            _selected['vectorizer'] = vectorizer

            print(f"Загружена модель: {model_dropdown.value}")
            print(f"Загружен векторизатор: {vectorizer_dropdown.value}")

    load_button.on_click(on_button_click)

    display(model_dropdown, vectorizer_dropdown, load_button, output)

def get_selected_model_and_vectorizer():
    """
    Возвращает выбранные модель и векторизатор.
    """
    model = _selected['model']
    vectorizer = _selected['vectorizer']
    if model is None or vectorizer is None:
        raise RuntimeError("Модель или векторизатор не выбраны. Сначала вызовите select_model_interactively().")
    return model, vectorizer

def clean_data(df):
    """
    Удаляет ненужные новости типа дайджестов.
    """
    return df[~df['Text'].str.contains(
        "дайджест|краткий обзор экономической повестки дня",
        case=False, na=False
    )]


def detect_signals(df_news, vectorizer, model, log_path='market_analyzer/data/logged_predictions.csv'):
    """
    Классифицирует все новости, добавляет колонку Signal (0/1),

    """
    df_news_clean = clean_data(df_news).copy()
    texts = df_news_clean['Text'].fillna("").tolist()
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    df_news_clean['Signal'] = preds
    df_news_clean['LoggedAt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    write_header = not os.path.exists(log_path)
    df_news_clean.to_csv(log_path, mode='a', header=write_header, index=False, encoding='utf-8')

    return df_news_clean[df_news_clean['Signal'] == 1]