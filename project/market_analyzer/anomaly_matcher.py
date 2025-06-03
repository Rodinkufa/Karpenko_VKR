import matplotlib.pyplot as plt
from moexalgo import Ticker
import pandas as pd

def analyze_volume_with_news(ticker_symbol, date, news_time):
    """
    Анализирует торговый объем и возвращает DataFrame с аномалиями до выхода новости.
    """
    try:
        ticker = Ticker(ticker_symbol)
        df_candles = ticker.candles(start=date, end=date, period='1min')
        df_candles['begin'] = pd.to_datetime(df_candles['begin'])

        mean_vol = df_candles['volume'].mean()
        std_vol = df_candles['volume'].std()
        threshold = mean_vol + 3 * std_vol

        outliers = df_candles[df_candles['volume'] > threshold]
        news_datetime = pd.to_datetime(f"{date} {news_time}")
        outliers_before = outliers[outliers['begin'] < news_datetime]

        plt.figure(figsize=(14, 6))
        plt.plot(df_candles['begin'], df_candles['volume'], label='Volume', alpha=0.6)
        plt.scatter(outliers['begin'], outliers['volume'], color='red', label='Outliers')
        plt.axhline(y=threshold, color='orange', linestyle='--', label='3σ Threshold')
        plt.axvline(x=news_datetime, color='green', linestyle='--', label='News Release')
        plt.title(f'Trading Volume for {ticker_symbol} on {date}')
        plt.xlabel('Time')
        plt.ylabel('Volume')
        plt.grid(True)
        plt.legend(loc='upper left', fontsize='small')
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.tight_layout()
        plt.show()

        df_result = outliers_before[['begin']].copy()
        df_result.rename(columns={'begin': 'Timestamp'}, inplace=True)
        df_result['Ticker'] = ticker_symbol
        df_result['NewsTime'] = news_datetime

        return df_result

    except Exception as e:
        print(f"❌ Error analyzing {ticker_symbol}: {e}")
        return pd.DataFrame(columns=['Timestamp', 'Ticker', 'NewsTime'])


def match_all_news(df_with_tickers, ticker_column='Ticker', date_column='Date'):
    """
    Обрабатывает DataFrame с новостями и тикерами.
    Возвращает DataFrame с аномалиями до новостей.
    """
    all_anomalies_before_news = []

    for idx, row in df_with_tickers.iterrows():
        ticker_symbol = row[ticker_column]
        if ticker_symbol == 'N/A' or pd.isna(ticker_symbol):
            continue

        full_datetime = pd.to_datetime(row[date_column])
        date_str = full_datetime.strftime('%Y-%m-%d')
        time_str = full_datetime.strftime('%H:%M:%S')

        df_anomalies = analyze_volume_with_news(ticker_symbol, date_str, time_str)
        all_anomalies_before_news.append(df_anomalies)

    df_all = pd.concat(all_anomalies_before_news, ignore_index=True)
    return df_all
