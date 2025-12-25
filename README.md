# Fake-News-Detection-Using-Machine-Learning-and-NLP

Fake News Detection Using NLP and Machine Learning is a Python-based project that implements a supervised machine learning pipeline to classify news articles as **Fake** or **Real** using Natural Language Processing techniques. The system performs data cleaning, text preprocessing including lowercasing, tokenization, stopword removal, and lemmatization, and feature extraction using TF-IDF vectorization. Three machine learning models—Logistic Regression, Multinomial Naïve Bayes, and Random Forest—are trained and evaluated using a standard train–test split approach. Model performance is assessed using accuracy scores, confusion matrices, and classification reports, with Logistic Regression achieving the highest accuracy of **94%**. The project is implemented in Python using libraries such as pandas, NumPy, scikit-learn, NLTK, and spaCy, and is developed in a Jupyter Notebook environment for reproducibility and academic analysis. This repository is intended for educational and research purposes and demonstrates a practical application of machine learning techniques for fake news detection.

---

## 📌 Project Overview

The objective of this project is to evaluate the effectiveness of different supervised machine learning models in detecting fake news. The system uses TF-IDF feature representation and compares the performance of three widely used classifiers:

- Logistic Regression
- Multinomial Naïve Bayes
- Random Forest

The project is implemented in Python using standard machine learning and NLP libraries and is developed as part of an academic study.

---

## 🛠️ Technologies and Tools Used

- **Programming Language:** Python
- **Development Environment:** Jupyter Notebook

**Libraries:**
- pandas
- numpy
- scikit-learn
- nltk
- spaCy
- matplotlib (for basic visualization)

---
Fake-News-Detection/
│
├── Fake News Detection using NLP.ipynb # Main implementation notebook
├── README.md # Project documentation
├── dataset/ # Fake and real news datasets
│ └── fakeorreal.csv
└── requirements.txt # Required Python libraries


---

## 📊 Dataset Description

The dataset consists of labeled news articles collected from publicly available sources. Each record contains:

- News article text
- A binary label indicating **Fake** or **Real**

The datasets are merged into a single labeled dataset and then split into training and testing sets for supervised learning.

---

## ⚙️ Implementation Workflow

### Data Loading and Cleaning
- Removal of missing values and duplicate entries

### Text Preprocessing
- Lowercasing
- Removal of punctuation and special characters
- Tokenization
- Stopword removal
- Lemmatization

### Feature Extraction
- TF-IDF Vectorization using scikit-learn

### Model Training
- Logistic Regression
- Multinomial Naïve Bayes
- Random Forest

### Model Evaluation
- Accuracy score
- Confusion matrix
- Classification report

---

## 📈 Results Summary

The experimental results show that Logistic Regression achieved the highest accuracy:

| Model                   | Accuracy |
|------------------------|----------|
| Logistic Regression     | 94%      |
| Multinomial Naïve Bayes | 91%      |
| Random Forest           | 91%      |

Logistic Regression performed best due to its effectiveness in handling high-dimensional sparse TF-IDF features.

---

## ▶️ How to Run the Project

Clone the repository:

```bash
git clone https://github.com/your-username/fake-news-detection.git


## 📂 Project Structure

