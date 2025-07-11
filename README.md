# 🧠 Fake News Detection Web Application

A modern web application built with Streamlit to showcase my machine learning model for fake news detection.

## 🚀 Features

- **Single Article Analysis**: Paste any news article and get instant predictions
- **Batch Analysis**: Upload CSV files with multiple articles for bulk processing
- **Confidence Scoring**: Shows confidence (if implemented with decision_function or probabilities)
- **Beautiful UI**: Modern, responsive design with optional animations
- **Clean Sidebar**: Tech stack, links, and about section

## 📋 Prerequisites

- Python 3.8 or higher
- Your `fake_news_model.pkl` file in the project directory


#DATASET USED : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data?select=True.csv

## 🛠️ Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your model file is in the correct location**:
   - Make sure `fake_news_model.pkl` is in the same directory as the app

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### How to Use

1. **Paste your news article** in the text area
2. Click "🚀 Predict"
3. The app will show whether it's 🟢 Real or 🔴 Fake
4. Check the prediction timestamp

## 📁 Project Structure

```
fake-news-app/
├── app.py                 # Streamlit application
├── fake_news_model.pkl    # Your trained ML model
├── requirements.txt       # Python dependencies
├── README.md              # This file
```


## 🔧 Troubleshooting

### Common Issues

1. **Model not loading**:
   - Ensure `fake_news_model.pkl` is in the correct directory

2. **Import errors**:
   - Run `pip install -r requirements.txt`

3. **Port already in use**:
   - Run with another port:
     ```bash
     streamlit run app.py --server.port 8502
     ```

## 🤝 Contributing

Feel free to customize or enhance this app:
- Add SHAP or LIME explainability
- Include batch prediction
- Deploy to Hugging Face or AWS

## 📄 License

This project is open source and available under the MIT License.
---

**Built with ❤️ using Streamlit and scikit-learn**
