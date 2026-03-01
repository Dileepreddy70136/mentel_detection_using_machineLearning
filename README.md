Mental Detection Using Machine Learning

A Machine Learning based web application that analyzes user text input and predicts mental state such as stress, anxiety, or depression level. The system generates a wellness score and provides basic suggestions based on prediction results.

Project Overview

This project uses Natural Language Processing (NLP) techniques to process user-written text (journal entries or messages) and classify the mental state using a trained machine learning model.

The goal of this system is to demonstrate practical implementation of:

Text preprocessing

Feature extraction

Model training and prediction

Web deployment using Flask

Features

Text-based mental state prediction

Wellness score calculation

Model confidence percentage

Suggestion system based on prediction

Clean and responsive user interface

Real-time result generation

Technologies Used

Python

Scikit-learn

Pandas

NumPy

Flask

HTML

CSS

JavaScript

System Workflow

User enters text input

Text preprocessing (cleaning, tokenization, stopword removal)

Feature extraction using TF-IDF / CountVectorizer

Machine Learning model prediction

Display of:

Mental State

Wellness Score

Confidence Score

Recommended Suggestions

Project Structure
mentel_detection_using_machineLearning/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
Installation and Setup
Clone Repository
git clone https://github.com/yourusername/mentel_detection_using_machineLearning.git
Navigate to Project Folder
cd mentel_detection_using_machineLearning
Install Dependencies
pip install -r requirements.txt
Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000
Sample Output

Mental State: Stressed / Anxious
Wellness Score: 30%
Model Confidence: 33.7%

Suggestions:

Practice controlled breathing

Take short breaks from screen time

Communicate with trusted individuals

Future Improvements

Deep Learning model integration

Improved dataset for higher accuracy

Graph-based emotion visualization

Database integration

Mobile-first UI enhancement

Author

Dileep Kumar Suggu
