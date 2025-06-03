from setuptools import setup, find_packages

setup(
    name='market_analyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'nest_asyncio',
        'telethon',
        'xgboost',
        'joblib',
        'scikit-learn',
        'matplotlib',
        'natasha',
        'moexalgo',
        'openpyxl'
    ],
    include_package_data=True,
    author='RodinK_UFA',
    description='Library for detecting market anomalies using Telegram news and MOEX data',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
