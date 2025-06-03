import joblib
import xgboost as xgb
import pandas as pd

def load_vectorizer(path='market_analyzer/models/tfidf_vectorizer.joblib'):
    return joblib.load(path)

def load_model(path='market_analyzer/models/xgb_best_weighted_model.json'):
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model

def clean_data(df):
    """
    Удаляет ненужные новости типа дайджестов.
    """
    filtered_df = df[~df['Text'].str.contains(
        "дайджест|краткий обзор экономической повестки дня",
        case=False, na=False
    )]
    return filtered_df

def detect_signals(df_news, vectorizer, model):
    """
    Классифицирует новости как сигнальные.
    """
    df_news_clean = clean_data(df_news)
    texts = df_news_clean['Text'].fillna("").tolist()
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    df_news_clean['Signal'] = preds
    return df_news_clean[df_news_clean['Signal'] == 1]
