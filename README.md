📰 Project Overview

This project is a Fake News Prediction System built using Machine Learning and Natural Language Processing (NLP) techniques. The model analyzes news articles and predicts whether the news is Real or Fake.

The project uses:

Python
Pandas
NumPy
NLTK
Scikit-learn
TF-IDF Vectorization
Logistic Regression

This project demonstrates text preprocessing, feature extraction, model training, and prediction using machine learning algorithms.

🚀 Features
Data preprocessing and cleaning
Stopword removal
Stemming using Porter Stemmer
TF-IDF Vectorization
Machine Learning model training
Fake vs Real news prediction
Accuracy evaluation
Simple and beginner-friendly implementation
🛠️ Technologies Used
Technology	Purpose
Python	Programming Language
Pandas	Data handling
NumPy	Numerical operations
NLTK	Text preprocessing
Scikit-learn	Machine learning
TF-IDF Vectorizer	Feature extraction
Logistic Regression	Classification model
Jupyter Notebook	Development environment
📂 Dataset

The dataset contains:

News Title
Author
News Content
Label
Labels
0 → Real News
1 → Fake News

Dataset used for training and testing the machine learning model.

⚙️ Project Workflow
1. Import Libraries

Import all required Python libraries.

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
2. Data Preprocessing
Remove special characters
Convert text to lowercase
Remove stopwords
Perform stemming
Stemming Function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
3. Feature Extraction

Convert text data into numerical vectors using TF-IDF.

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
4. Train Test Split

Split the dataset into training and testing data.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
5. Model Training

Train the Logistic Regression model.

model = LogisticRegression()
model.fit(X_train, Y_train)
6. Model Evaluation

Evaluate the accuracy of the model.

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
📈 Model Accuracy

The model achieves good accuracy on both training and testing datasets.

Example:

Training Accuracy: ~98%
Testing Accuracy: ~96%

(Accuracy may vary depending on dataset and preprocessing.)

🧠 Machine Learning Concepts Used
Natural Language Processing (NLP)
TF-IDF Vectorization
Logistic Regression
Text Classification
Data Cleaning
Feature Engineering
▶️ How to Run the Project
Step 1: Clone Repository
git clone https://github.com/your-username/fake-news-prediction.git
Step 2: Install Required Libraries
pip install numpy pandas nltk scikit-learn
Step 3: Run Jupyter Notebook
jupyter notebook

Open the notebook and run all cells.

📸 Sample Prediction
input_data = X_test[0]
prediction = model.predict(input_data)


if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
📁 Project Structure
Fake-News-Prediction/
│
├── Fake News Prediction using Machine Learning.ipynb
├── train.csv
├── README.md
└── requirements.txt
🔮 Future Improvements
Deploy using Flask or Streamlit
Add Deep Learning models
Improve dataset size
Build a web application
Add live news prediction
🤝 Contributing

Contributions are welcome.

Fork the repository
Create a new branch
Commit changes
Push the branch
Create a Pull Request
📜 License

This project is open-source and available under the MIT License.

👨‍💻 Author

Anjith Shetty

Machine Learning & Data Science Enthusiast
