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
import xgboost as xgb
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# ML Models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDClassifier
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, roc_auc_score, confusion_matrix
)

class EnhancedMLChatbot:
    def __init__(self):
        self.data = None
        self.model = None
        self.target = None
        self.model_pipeline = None
        self.feature_importance = None
        self.last_predictions = None
        
        # Available models
        self.regression_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'elastic': ElasticNet(),
            'rf': RandomForestRegressor(),
            'xgboost': XGBRegressor(),
            'lightgbm': LGBMRegressor(),
            'gbm': GradientBoostingRegressor(),
            'svr': SVR(),
            'knn': KNeighborsRegressor(),
            'mlp': MLPRegressor(),
            'dt': DecisionTreeRegressor(),
            'adaboost': AdaBoostRegressor()
        }
        
        self.classification_models = {
            'logistic': LogisticRegression(),
            'rf': RandomForestClassifier(),
            'xgboost': XGBClassifier(),
            'lightgbm': LGBMClassifier(),
            'gbm': GradientBoostingClassifier(),
            'svc': SVC(probability=True),
            'knn': KNeighborsClassifier(),
            'mlp': MLPClassifier(),
            'dt': DecisionTreeClassifier(),
            'adaboost': AdaBoostClassifier(),
            'sgd': SGDClassifier(loss='modified_huber')
        }

    def preprocess_query(self, query):
        """Advanced query preprocessing"""
        # Tokenization
        tokens = word_tokenize(query.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # POS tagging
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract key information using spaCy
        doc = nlp(query)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'entities': entities,
            'doc': doc
        }

    def extract_model_params(self, query_info):
        """Extract model parameters from query"""
        params = {}
        text = ' '.join(query_info['tokens'])
        
        # Extract numerical parameters
        numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
        
        if 'estimator' in text or 'trees' in text:
            params['n_estimators'] = int(numbers[0]) if numbers else 100
            
        if 'depth' in text:
            params['max_depth'] = int(numbers[0]) if numbers else None
            
        if 'neighbor' in text:
            params['n_neighbors'] = int(numbers[0]) if numbers else 5
            
        return params

    def create_model_pipeline(self, X, model_type='auto'):
        """Create preprocessing pipeline"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer()),
            ('scaler', RobustScaler())
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
        
        return preprocessor

    def interpret_ml_query(self, query):
        """Interpret ML-related queries"""
        query_info = self.preprocess_query(query)
        tokens = query_info['tokens']
        
        # Detect intent
        if any(word in tokens for word in ['train', 'create', 'build']):
            return 'train', self.extract_model_params(query_info)
        
        if any(word in tokens for word in ['predict', 'forecast', 'estimate']):
            return 'predict', None
            
        if any(word in tokens for word in ['evaluate', 'performance', 'score']):
            return 'evaluate', None
            
        if any(word in tokens for word in ['explain', 'interpret', 'importance']):
            return 'explain', None
            
        return 'unknown', None

    def train_model(self, data, target_col, model_type='auto', params=None):
        """Train ML model with advanced preprocessing"""
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            
            # Determine if classification or regression
            if model_type == 'auto':
                if len(np.unique(y)) < 10 or y.dtype == 'object':
                    model_type = 'classification'
                else:
                    model_type = 'regression'
            
            # Create preprocessing pipeline
            preprocessor = self.create_model_pipeline(X)
            
            # Select model
            if model_type == 'classification':
                model = self.classification_models.get(params.get('model', 'rf'))
            else:
                model = self.regression_models.get(params.get('model', 'rf'))
            
            # Update model parameters
            if params:
                model.set_params(**params)
            
            # Create full pipeline
            self.model_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model_pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_pipeline.predict(X_test)
            
            if model_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                results = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                # ROC AUC for binary classification
                if len(np.unique(y)) == 2:
                    y_prob = self.model_pipeline.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_test, y_prob)
                
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2_score': r2
                }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            return results
            
        except Exception as e:
            return f"Error training model: {str(e)}"

    def process_query(self, query):
        """Enhanced query processing"""
        try:
            # Preprocess query
            query_info = self.preprocess_query(query)
            
            # Extract intent and parameters
            intent, params = self.interpret_ml_query(query)
            
            if intent == 'train':
                # Extract target variable
                target_col = None
                for token, pos in query_info['pos_tags']:
                    if pos.startswith('NN') and token in st.session_state.data.columns:
                        target_col = token
                        break
                
                if not target_col:
                    return "Please specify which column you want to predict."
                
                # Train model
                results = self.train_model(st.session_state.data, target_col, params=params)
                
                # Format results
                if isinstance(results, dict):
                    response = "Model trained successfully!\n\n"
                    for metric, value in results.items():
                        if isinstance(value, (int, float)):
                            response += f"{metric}: {value:.4f}\n"
                    
                    if self.feature_importance is not None:
                        response += "\nTop 5 Important Features:\n"
                        for _, row in self.feature_importance.head().iterrows():
                            response += f"- {row['feature']}: {row['importance']:.4f}\n"
                    
                    return response
                else:
                    return results
            
            elif intent == 'predict':
                if self.model_pipeline is None:
                    return "Please train a model first!"
                
                # Create input interface for predictions
                st.write("Enter values for prediction:")
                input_data = {}
                for col in st.session_state.data.columns:
                    if col != self.target:
                        input_data[col] = st.text_input(f"Enter {col}")
                
                if st.button("Predict"):
                    try:
                        input_df = pd.DataFrame([input_data])
                        prediction = self.model_pipeline.predict(input_df)
                        self.last_predictions = prediction
                        return f"Prediction: {prediction[0]}"
                    except Exception as e:
                        return f"Error making prediction: {str(e)}"
            
            elif intent == 'evaluate':
                if self.model_pipeline is None:
                    return "Please train a model first!"
                
                # Perform cross-validation
                scores = cross_val_score(self.model_pipeline, 
                                      st.session_state.data.drop(self.target, axis=1),
                                      st.session_state.data[self.target],
                                      cv=5)
                
                return f"Cross-validation scores:\nMean: {scores.mean():.4f}\nStd: {scores.std():.4f}"
            
            elif intent == 'explain':
                if self.feature_importance is None:
                    return "No feature importance available. Please train a model that supports feature importance."
                
                # Create feature importance plot
                fig = px.bar(self.feature_importance,
                           x='feature',
                           y='importance',
                           title='Feature Importance')
                st.plotly_chart(fig)
                
                return "Feature importance plot generated!"
            
            else:
                return "I'm not sure how to help with that. Try asking about training a model, making predictions, or evaluating model performance."
                
        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    st.title("🤖 Enhanced ML Analysis Chatbot")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = EnhancedMLChatbot()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success(f"Dataset loaded successfully! Shape: {data.shape}")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
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
        response = st.session_state.chatbot.process_query(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
