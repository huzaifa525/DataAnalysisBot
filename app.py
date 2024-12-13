import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from transformers import pipeline
from ctgan import CTGAN
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import pytesseract
from PIL import Image

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

@st.cache_data
def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    data = data.drop_duplicates()  # Remove duplicates
    return data

class MLChatbot:
    def __init__(self):
        self.data = None
        self.model = None
        self.target = None
        self.model_pipeline = None
        self.text_generator = pipeline("text-generation", model="gpt2")

        # Available models
        self.regression_models = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': XGBRegressor(),
            'svr': SVR(),
            'mlp': MLPRegressor(max_iter=1000),
            'dt': DecisionTreeRegressor()
        }

        self.classification_models = {
            'logistic': LogisticRegression(),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': XGBClassifier(),
            'svc': SVC(probability=True),
            'mlp': MLPClassifier(max_iter=1000),
            'dt': DecisionTreeClassifier()
        }

    def train_model(self, data, target_col, model_type):
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            self.target = target_col

            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            if model_type == 'classification':
                base_model = self.classification_models['xgboost']
            else:
                base_model = self.regression_models['xgboost']

            self.model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', base_model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model_pipeline.fit(X_train, y_train)

            y_pred = self.model_pipeline.predict(X_test)

            if model_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                return f"Model trained successfully!\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}"

            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                return f"Model trained successfully!\nRMSE: {np.sqrt(mse):.4f}\nR² Score: {r2:.4f}"

        except Exception as e:
            return f"Error training model: {str(e)}"

    def analyze_distribution(self, data, column_query):
        try:
            matching_cols = [col for col in data.columns if column_query.lower() in col.lower()]

            if not matching_cols:
                return f"Column containing '{column_query}' not found. Available columns: {', '.join(data.columns)}"

            col = matching_cols[0]
            value_counts = data[col].value_counts()

            response = f"\nDistribution of {col}:\n"
            for value, count in value_counts.items():
                percentage = (count / len(data)) * 100
                response += f"- {value}: {count} ({percentage:.1f}%)\n"

            response += f"\nTotal unique values: {len(value_counts)}"
            return response

        except Exception as e:
            return f"Error analyzing distribution: {str(e)}"

    def create_synthetic_data(self, data):
        try:
            discrete_columns = data.select_dtypes(include=['object']).columns.tolist()
            ctgan = CTGAN()
            ctgan.fit(data, discrete_columns=discrete_columns)
            synthetic_data = ctgan.sample(len(data))
            return synthetic_data
        except Exception as e:
            return f"Error creating synthetic data: {str(e)}"

    def generate_text_insights(self, prompt):
        try:
            result = self.text_generator(prompt, max_length=50, num_return_sequences=1)
            return result[0]['generated_text']
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def enhance_data_quality(self, data):
        try:
            for col in data.select_dtypes(include=['float64', 'int64']).columns:
                data[col].fillna(data[col].median(), inplace=True)
            return "Data quality enhanced with missing values handled."
        except Exception as e:
            return f"Error enhancing data quality: {str(e)}"

# Streamlit App
st.title("ML Analysis Chatbot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv', 'xlsx'])
if uploaded_file:
    data = load_data(uploaded_file)
    data = preprocess_data(data)
    chatbot = MLChatbot()

    st.sidebar.subheader("Data Overview")
    if st.sidebar.checkbox("Show Data Info"):
        chatbot.show_data_info(data)

    if st.sidebar.checkbox("Analyze Column"):
        column = st.sidebar.selectbox("Select Column", options=data.columns)
        chatbot.analyze_distribution(data, column)

    if st.sidebar.checkbox("Correlation Analysis"):
        chatbot.perform_correlation_analysis(data)

    if st.sidebar.checkbox("Create Synthetic Data"):
        synthetic_data = chatbot.create_synthetic_data(data)
        st.write(synthetic_data)

    if st.sidebar.checkbox("Generate Text Insights"):
        prompt = st.text_input("Enter prompt for insights:")
        if prompt:
            insights = chatbot.generate_text_insights(prompt)
            st.text(insights)

    if st.sidebar.checkbox("Enhance Data Quality"):
        st.text(chatbot.enhance_data_quality(data))

    st.sidebar.subheader("Train Model")
    target_col = st.sidebar.selectbox("Select Target Column", options=data.columns)
    model_type = st.sidebar.radio("Model Type", ['classification', 'regression'])
    if st.sidebar.button("Train Model"):
        result = chatbot.train_model(data, target_col, model_type)
        st.success(result)

# Chat Interface
st.header("Chat with your data!")
for message in st.session_state['messages']:
    st.write(message)

user_input = st.text_input("Ask about your data:")
if user_input:
    st.session_state['messages'].append(f"User: {user_input}")
    response = chatbot.analyze_distribution(data, user_input)
    st.session_state['messages'].append(f"Chatbot: {response}")
    st.write(f"Chatbot: {response}")
