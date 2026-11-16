# SentimentX - Ecommerce Product Sentiment Analysis

![Stars](https://img.shields.io/github/stars/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms?style=for-the-badge&logo=github&color=yellow) ![Forks](https://img.shields.io/github/forks/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms?style=for-the-badge&logo=github) ![Last commit](https://img.shields.io/github/last-commit/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms?style=for-the-badge&logo=git) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![CSS](https://img.shields.io/badge/css3-1572B6?style=for-the-badge&logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E) ![FastAPI](https://img.shields.io/badge/fastapi-005571?style=for-the-badge&logo=fastapi&logoColor=white)

## Problem
SentimentX is a project that uses machine learning algorithms to analyze product reviews for sentiment analysis. The goal is to provide a user-friendly interface for users to easily upload their own product reviews and get a sentiment analysis report.

## Features

### 1. CSV Upload
- Users can upload a CSV file containing product reviews.
- SentimentX labels each review as positive, neutral, or negative, showing where it sits on a sentiment scale.

### 2. Metrics
- The project provides a summary of the sentiment analysis results, including accuracy, macro F1 score, and test reviews.

### 3. Training
- SentimentX is trained on a large labelled dataset of product reviews.
- The model is trained using the `make_dataset.py` script to generate a large labelled review corpus.

### 4. Data Processing
- The project uses Python and FastAPI to handle data processing and API requests.

## Technologies
- Python 37.9%
- FastAPI
- pandas
- scikit-learn
- python-multipart

## Project Structure
- **artifacts/**: Contains the model and metrics.
- **data/**: Contains the CSV file for product reviews.
- **static/**: Contains CSS, JavaScript, and other static assets.
- **app.py**: Application entry point.
- **make_dataset.py**: Generates a large labelled review corpus.
- **model.py**: SentimentX serving layer.
- **train.py**: Train the SentimentX model on a labelled review CSV.

## Quick Start
1. Clone the repository: `git clone https://github.com/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `uvicorn app:app --reload`

## How to Use
1. Upload a CSV file containing product reviews.
2. The model analyzes the reviews and provides a sentiment analysis report.

## How It Works
- SentimentX uses a multinomial Logistic Regression model to assign probabilities to the three classes.
- The model's accuracy is the highest probability, and the score is P(positive) minus P(negative).
- Metrics include accuracy, macro F1 score, and test reviews.

## Dependencies

- `fastapi`
- `uvicorn`
- `pandas`
- `scikit-learn`
- `python-multipart`

## Prerequisites

Install these first:

- Git
- Python 3 and pip

## Support

If this project helps you, give it a star on [GitHub](https://github.com/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms) and share it with others. Found a problem? [Open an issue](https://github.com/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms/issues).

## Repository

https://github.com/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms

## Contributing
Feel free to contribute to the project by adding new features or fixing bugs. Just clone the repository and make changes in the `app.py` file.

## License
The project is licensed under the MIT License. You can find the license details in the `LICENSE` file.

## How to Contribute
1. Fork the repository: `git clone https://github.com/PranavN-AI/SentimentX---Ecommerce-product-sentiment-analysis-using-Machine-Learning-Algorithms.git`
2. Make changes in the `app.py` file.
3. Commit your changes: `git commit -m "Add feature: sentiment analysis"`
4. Push your changes: `git push origin main`
5. Create a pull request: `git pull origin main`
