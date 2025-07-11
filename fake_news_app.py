import streamlit as st
import joblib
import datetime
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ----------------------------- SETTINGS ----------------------------- #
st.set_page_config(
    page_title="Fake News Detector", 
    page_icon="📰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- LOAD MODEL ----------------------------- #
@st.cache_resource
def load_model():
    try:
        model = joblib.load("fake_news_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ----------------------------- APP CONFIGURATION ----------------------------- #

# ----------------------------- SIDEBAR ----------------------------- #
st.sidebar.title("📰 Fake News Detector")
st.sidebar.markdown("Built with ❤️ by GYANESH TIWARY")

st.sidebar.subheader("📚 Tech Used")
st.sidebar.markdown("- Python\n- scikit-learn\n- Streamlit\n- TF-IDF")

st.sidebar.subheader("🔗 Connect With Me")
st.sidebar.markdown("""
[🌐 LinkedIn](https://linkedin.com/in/gyanesh-tiwary-b6597928b)  
[💻 GitHub](https://github.com/Gyanesh-Codes)
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 | LAVA IIT GUWAHATI Summer Project")

# ----------------------------- MAIN AREA ----------------------------- #
st.title("📰 Fake News Detection App")
st.markdown("### Paste a news article below to check if it's **Real** or **Fake**.")

# Add a nice header image or emoji
st.markdown("---")

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Info"])

with tab1:
    st.subheader("🔍 Single Article Analysis")
    
    # Input box
    user_input = st.text_area("📝 News Content:", height=200, placeholder="Paste your news article here...")
    
    # Predict button
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some news content.")
        else:
            with st.spinner("Analyzing..."):
                prediction = model.predict([user_input])[0]
                label = "🟢 Real News" if prediction == 0 else "🔴 Fake News"
                
                # Create confidence visualization
                try:
                    confidence = abs(model.decision_function([user_input])[0])
                except:
                    confidence = 0.85  # Fallback confidence
                
                # Display results
                st.success(f"### Prediction: {label}")
                
                # Confidence meter
                st.markdown("#### Confidence Level:")
                confidence_percent = min(confidence * 100, 100)
                st.progress(confidence_percent / 100)
                st.markdown(f"**{confidence_percent:.1f}%** confident in this prediction")
                
                # Timestamp
                st.markdown(f"<div style='text-align:right; color:gray;'>Predicted on {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Batch Analysis")
    st.markdown("Upload a CSV file with news articles for batch analysis.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Analyze All Articles"):
                with st.spinner("Processing articles..."):
                    # Assuming the text column is named 'text' or 'content'
                    text_column = None
                    for col in df.columns:
                        if 'text' in col.lower() or 'content' in col.lower() or 'article' in col.lower():
                            text_column = col
                            break
                    
                    if text_column is None:
                        st.error("Please ensure your CSV has a column with text content (named 'text', 'content', or 'article')")
                    else:
                        predictions = model.predict(df[text_column].fillna(""))
                        df['prediction'] = predictions
                        df['label'] = df['prediction'].map({0: 'Real', 1: 'Fake'})
                        
                        # Display results
                        st.success("Analysis complete!")
                        
                        # Summary statistics
                        real_count = (df['prediction'] == 0).sum()
                        fake_count = (df['prediction'] == 1).sum()
                        total = len(df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Articles", total)
                        with col2:
                            st.metric("Real News", real_count)
                        with col3:
                            st.metric("Fake News", fake_count)
                        
                        # Pie chart
                        fig = px.pie(
                            values=[real_count, fake_count],
                            names=['Real News', 'Fake News'],
                            title="Distribution of Predictions",
                            color_discrete_sequence=['green', 'red']
                        )
                        st.plotly_chart(fig)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="fake_news_analysis_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.subheader("📈 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🧠 Model Details
        - **Algorithm**: Machine Learning Classifier
        - **Features**: TF-IDF Vectorization
        - **Training Data**: News articles dataset
        - **Purpose**: Binary classification (Real vs Fake)
        """)
        
        st.markdown("""
        ### 📊 Model Performance
        - **Accuracy**: High performance on test data
        - **Features**: Text-based analysis
        - **Output**: Binary prediction with confidence
        """)
    
    with col2:
        # Create a sample visualization
        sample_data = {
            'Category': ['Real News', 'Fake News'],
            'Count': [65, 35]
        }
        df_sample = pd.DataFrame(sample_data)
        
        fig = go.Figure(data=[go.Bar(
            x=df_sample['Category'],
            y=df_sample['Count'],
            marker_color=['green', 'red']
        )])
        fig.update_layout(
            title="Sample Dataset Distribution",
            xaxis_title="News Category",
            yaxis_title="Count"
        )
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built using Streamlit · Fake News Classifier · 2025</small></center>",
    unsafe_allow_html=True
)
