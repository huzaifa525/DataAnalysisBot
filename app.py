import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_auc_score, confusion_matrix
)
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Machine Learning Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

class MLChatbot:
    def __init__(self):
        self.data = None
        self.model = None
        self.target = None
        self.model_pipeline = None
        self.feature_importance = None
        
        # Available models
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'rf': RandomForestRegressor(),
            'xgboost': XGBRegressor(),
            'svr': SVR(),
            'mlp': MLPRegressor(),
            'dt': DecisionTreeRegressor()
        }
        
        self.classification_models = {
            'logistic': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'xgboost': XGBClassifier(),
            'svc': SVC(probability=True),
            'mlp': MLPClassifier(),
            'dt': DecisionTreeClassifier()
        }

    def process_query(self, query):
        """Process natural language queries"""
        query = query.lower()
        tokens = word_tokenize(query)
        
        try:
            # Data loading queries
            if any(word in tokens for word in ['load', 'upload', 'import']):
                return self.handle_data_upload()
            
            if st.session_state.data is None:
                return "Please upload a dataset first! You can say 'load data' to upload a file."
            
            # Show data info
            if 'show' in tokens and ('data' in tokens or 'info' in tokens):
                return self.show_data_info()
            
            # Handle ML queries
            if any(word in tokens for word in ['train', 'predict', 'model']):
                return self.handle_ml_query(query)
            
            # Handle visualization queries
            if any(word in tokens for word in ['plot', 'chart', 'graph', 'show']):
                return self.handle_visualization(query)
            
            # Handle analysis queries
            if any(word in tokens for word in ['analyze', 'analyse', 'statistics']):
                return self.analyze_data(query)
            
            return ("I can help you with:\n"
                   "1. Loading data ('load data')\n"
                   "2. Showing data info ('show info')\n"
                   "3. Training models ('train model to predict X')\n"
                   "4. Creating visualizations ('show chart of X')\n"
                   "5. Analyzing data ('analyze X')")
            
        except Exception as e:
            return f"Error: {str(e)}"

    def handle_data_upload(self):
        """Handle data upload request"""
        if st.session_state.data is not None:
            data = st.session_state.data
            return (f"Current dataset info:\n"
                   f"- Shape: {data.shape}\n"
                   f"- Columns: {', '.join(data.columns)}\n\n"
                   f"You can start analyzing this data!")
        return "Please use the sidebar to upload your data file (CSV or Excel)."

    def handle_ml_query(self, query):
        """Handle machine learning queries"""
        tokens = word_tokenize(query.lower())
        
        if 'train' in tokens:
            # Extract target variable
            target_index = tokens.index('predict') + 1 if 'predict' in tokens else -1
            if target_index >= 0 and target_index < len(tokens):
                target = tokens[target_index]
                
                # Find actual column name (case-insensitive)
                target_col = None
                for col in st.session_state.data.columns:
                    if col.lower() == target:
                        target_col = col
                        break
                
                if target_col is None:
                    return f"Column '{target}' not found. Available columns: {', '.join(st.session_state.data.columns)}"
                
                # Train model
                return self.train_model(st.session_state.data, target_col)
            
            return "Please specify what to predict. Example: 'train model to predict price'"
            
        elif 'predict' in tokens and self.model_pipeline:
            # Create prediction interface
            st.write("Enter values for prediction:")
            input_data = {}
            for col in st.session_state.data.columns:
                if col != self.target:
                    input_data[col] = st.text_input(f"Enter {col}")
            
            if st.button("Predict"):
                try:
                    input_df = pd.DataFrame([input_data])
                    prediction = self.model_pipeline.predict(input_df)
                    return f"Prediction: {prediction[0]}"
                except Exception as e:
                    return f"Error making prediction: {str(e)}"
        
        return "Please train a model first or specify what you want to predict."

    def train_model(self, data, target_col):
        """Train ML model with preprocessing"""
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            self.target = target_col
            
            # Determine if classification or regression
            if len(np.unique(y)) < 10 or y.dtype == 'object':
                model_type = 'classification'
                base_model = RandomForestClassifier()
            else:
                model_type = 'regression'
                base_model = RandomForestRegressor()
            
            # Create preprocessing pipeline
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', LabelEncoder())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Create and train pipeline
            self.model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', base_model)
            ])
            
            # Split data and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model_pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_pipeline.predict(X_test)
            
            if model_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                response = (f"Model trained successfully!\n\n"
                          f"Accuracy: {accuracy:.4f}\n"
                          f"F1 Score: {f1:.4f}")
                
                # Feature importance
                if hasattr(base_model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': base_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    response += "\n\nTop 5 Important Features:\n"
                    for _, row in importance.head().iterrows():
                        response += f"- {row['feature']}: {row['importance']:.4f}\n"
                
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                response = (f"Model trained successfully!\n\n"
                          f"RMSE: {np.sqrt(mse):.4f}\n"
                          f"R² Score: {r2:.4f}")
            
            return response
            
        except Exception as e:
            return f"Error training model: {str(e)}"

    def handle_visualization(self, query):
        """Create visualizations based on query"""
        data = st.session_state.data
        
        # Extract column names from query
        columns = [col for col in data.columns if col.lower() in query.lower()]
        
        if not columns:
            return "Please specify which columns you want to visualize."
        
        if 'scatter' in query and len(columns) >= 2:
            fig = px.scatter(data, x=columns[0], y=columns[1], 
                           title=f"Scatter Plot: {columns[0]} vs {columns[1]}")
            st.plotly_chart(fig)
            return "Here's your scatter plot!"
            
        elif 'bar' in query and columns:
            if data[columns[0]].dtype in ['int64', 'float64']:
                fig = px.bar(data[columns[0]].value_counts().reset_index(),
                           x='index', y=columns[0],
                           title=f"Bar Plot of {columns[0]}")
            else:
                fig = px.bar(data[columns[0]].value_counts().reset_index(),
                           x='index', y='count',
                           title=f"Bar Plot of {columns[0]}")
            st.plotly_chart(fig)
            return "Here's your bar plot!"
            
        elif 'histogram' in query and columns:
            fig = px.histogram(data, x=columns[0],
                             title=f"Histogram of {columns[0]}")
            st.plotly_chart(fig)
            return "Here's your histogram!"
        
        return "Please specify the type of plot (scatter, bar, histogram) and the columns to visualize."

    def analyze_data(self, query):
        """Analyze data based on query"""
        data = st.session_state.data
        
        # Extract column names from query
        columns = [col for col in data.columns if col.lower() in query.lower()]
        
        if not columns:
            return "Please specify which columns you want to analyze."
        
        results = []
        for col in columns:
            if data[col].dtype in ['int64', 'float64']:
                stats = data[col].describe()
                results.append(f"\nStatistics for {col}:")
                results.append(f"Mean: {stats['mean']:.2f}")
                results.append(f"Std: {stats['std']:.2f}")
                results.append(f"Min: {stats['min']:.2f}")
                results.append(f"Max: {stats['max']:.2f}")
            else:
                value_counts = data[col].value_counts()
                results.append(f"\nValue counts for {col}:")
                for val, count in value_counts.items():
                    results.append(f"{val}: {count}")
        
        return "\n".join(results)

def main():
    st.title("🤖 ML Analysis Chatbot")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Sidebar for file upload and data info
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                # Show loading spinner
                with st.spinner('Loading data...'):
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                    
                    st.session_state.data = data
                    st.success(f"✅ File uploaded successfully!")
                    
                    # Show data info in sidebar
                    st.header("📊 Data Info")
                    st.write(f"Rows: {data.shape[0]}")
                    st.write(f"Columns: {data.shape[1]}")
                    
                    # Show sample of the data
                    st.write("Preview:")
                    st.dataframe(data.head(3), use_container_width=True)
                    
                    # Show column info
                    st.write("Columns:")
                    for col in data.columns:
                        st.write(f"- {col} ({data[col].dtype})")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Initialize chatbot
    chatbot = MLChatbot()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your data!"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get chatbot response
        response = chatbot.process_query(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
