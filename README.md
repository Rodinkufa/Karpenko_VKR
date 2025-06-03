# Karpenko_VKR

This repository contains materials for a Master's thesis focused on detecting signs of insider trading on the stock market through the analysis of market anomalies **prior** to the release of public news signals.

## Repository Structure

- **`learn and validation/`** — research on machine learning models and feature selection.
- **`parsing_inter_fax_data/`** — code for parsing raw news data from the Telegram channel [@ifax_go](https://t.me/ifax_go) over a six-month period.
- **`Testing the CB method.ipynb`** — implementation of tests based on the methodology outlined by the Central Bank of Russia (MR-5 guidelines).
- **`data.rar`** — archive containing manually labeled news samples (signal/non-signal + associated tickers).
- **`project/`** — demonstration pipeline. Launch `analyze.ipynb` to evaluate and visualize results.

## Quick Start

1. Clone the repository:
    ```bash
    git clone https://github.com/Rodinkufa/Karpenko_VKR.git
    cd Karpenko_VKR/project
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the demo notebook:
    ```bash
    jupyter notebook analyze.ipynb
    ```

## Technologies Used

- Python 3.10+
- scikit-learn, pandas, numpy
- HuggingFace Transformers (BERT, RuBERT)
- Telethon
- Jupyter Notebook

## Project Objective

To develop a model for **automated classification of financial news**, capable of identifying market anomalies that may indicate **potential insider trading activity before official announcements are made**.

## 📎 Contact

**Author:** Karpenko Rodion  
**Email:** roalkarpenko@yandex.ru
