# Fake News Detection

## Overview
This project is a **Fake News Detection** system built using **Python, Scikit-learn, and Streamlit**. It analyzes text data to determine whether a news article is **real or fake** using **Natural Language Processing (NLP)** techniques.

## Features
- **TF-IDF Vectorization** for text processing
- **Machine Learning model** (Logistic Regression, Naive Bayes, etc.) for classification
- **User-friendly Web App** using Streamlit
- **Real-time prediction** of fake news

## Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Pickle (for model storage)

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/Mohitarora2435/Fake-News-Detector.git
cd fake-news-detector
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

## Dataset
The dataset used for training is based on publicly available fake news datasets. You can add your own dataset in CSV format and modify `data_preprocessing.py` accordingly.

## Model Training
To train the model, run:
```bash
python train_model.py
```
This will save the trained model as `model.pkl`.

## Future Improvements
- Integrate Deep Learning (LSTMs, Transformers)
- Enhance UI/UX with more interactivity
- Deploy on cloud platforms

## Contributing
Feel free to contribute by opening issues and pull requests.

## License
This project is licensed under the **MIT License**.

---
### 🔗 Connect with Me
[GitHub](https://github.com/Mohitarora2435) | [LinkedIn](https://linkedin.com/in/mohitarora2435)

