
# Email Spam Detector

A machine learning project that classifies text messages as Spam or Not Spam using Natural Language Processing (NLP).

## Features

- Predicts whether a message is spam or legitimate.
- Uses TF-IDF to convert text into numerical features.
- Uses Multinomial Naive Bayes for classification.
- Provides a simple interactive interface using Streamlit.
- Displays model accuracy and a classification report.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Natural Language Processing (NLP)

## Project Structure

Email-Spam-Detector/
- app.py
- train_model.py
- spam_detector_model.pkl
- requirements.txt
- README.md
- .gitignore
- data/
  - spam.tsv

## Installation and Setup

1. Clone this repository.
2. Open the project folder in your terminal.
3. Create and activate a virtual environment.
4. Install the dependencies:

   pip install -r requirements.txt

5. Train the model if needed:

   python train_model.py

6. Start the application:

   streamlit run app.py

## Machine Learning Workflow

1. Load and prepare the dataset.
2. Split the data into training and testing sets.
3. Convert message text using TF-IDF.
4. Train a Multinomial Naive Bayes classifier.
5. Evaluate model performance.
6. Predict whether a new message is spam or legitimate.

## Dataset

The project uses the SMS Spam Collection dataset, containing spam and legitimate text messages.

## Disclaimer

Predictions are not guaranteed to be correct. Always verify suspicious messages and email links independently.
