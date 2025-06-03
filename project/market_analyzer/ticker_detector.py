import pandas as pd
import requests
from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)

def extract_companies_natasha(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    companies = [span.text for span in doc.spans if span.type == 'ORG']
    return companies

def fetch_moex_tickers():
    url = 'https://iss.moex.com/iss/engines/stock/markets/shares/securities.json'
    response = requests.get(url)
    data = response.json()
    securities = data['securities']['data']
    columns = data['securities']['columns']
    df = pd.DataFrame(securities, columns=columns)
    df_filtered = df[['SECID', 'SHORTNAME', 'SECNAME', 'LATNAME']].dropna()
    df_filtered = df_filtered[df_filtered['SECID'].apply(lambda x: len(str(x)) <= 6)]

    company_to_ticker = {}
    for _, row in df_filtered.iterrows():
        names = [row['SECNAME'], row['SHORTNAME'], row['LATNAME']]
        for name in names:
            company_to_ticker[name.lower()] = row['SECID']

    company_to_ticker.update({
        'иват': 'IVAT',
        'iva technologies': 'IVAT',
        'норникель': 'GMKN',
        'норильский никель': 'GMKN',
        'fix price': 'FIXP',
        'фикс прайс': 'FIXP',
        'смарттехгрупп': 'CARM',
        "о'кей": 'OKEY'
    })

    remove_words = ['холдинг', 'holding', 'интернешионал', 'international']
    cleaned = {}
    for key, value in company_to_ticker.items():
        cleaned_key = key
        for word in remove_words:
            cleaned_key = cleaned_key.replace(word, '').strip()
        cleaned[cleaned_key] = value

    return cleaned

company_to_ticker = fetch_moex_tickers()
company_names = list(company_to_ticker.keys())
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
tfidf_matrix = vectorizer.fit_transform(company_names)

def find_ticker_by_top3(companies):
    for company in companies[:3]:
        org_norm = company.lower()
        target_vector = vectorizer.transform([org_norm])
        similarities = cosine_similarity(target_vector, tfidf_matrix)[0]
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        if best_similarity >= 0.5:
            best_match_name = company_names[best_idx]
            return company_to_ticker[best_match_name]
    return None

def detect_tickers(df_signals):
    df = df_signals.copy()
    df['company_natasha'] = df['Text'].apply(extract_companies_natasha)

    df['company_natasha'] = df['company_natasha'].apply(
        lambda companies: ['Т-технологии' if ('Росбанк' in c or 'T-банк' in c) else c for c in companies]
    )

    df['Ticker'] = df['company_natasha'].apply(find_ticker_by_top3)
    return df
