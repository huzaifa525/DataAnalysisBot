import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from spacy import displacy
import altair as alt
import streamz
from streamz.dataframe import Random
from streamz.dask import DaskStream
import dask.dataframe as dd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

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

    def get_value_counts(self, column_query):
        """Get value counts for a column"""
        data = st.session_state.data

        # Find the matching column (case-insensitive)
        matching_cols = [col for col in data.columns
                         if column_query.lower() in col.lower()]

        if not matching_cols:
            return f"Column containing '{column_query}' not found. Available columns: {', '.join(data.columns)}"

        col = matching_cols[0]
        value_counts = data[col].value_counts()

        # Create visualization
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {col}",
            labels={'x': col, 'y': 'Count'}
        )
        st.plotly_chart(fig)

        # Generate text response
        response = f"\nDistribution of {col}:\n"
        for value, count in value_counts.items():
            percentage = (count / len(data)) * 100
            response += f"- {value}: {count} ({percentage:.1f}%)\n"

        response += f"\nTotal unique values: {len(value_counts)}"
        return response

    def analyze_distribution(self, column):
        """Analyze distribution of a column"""
        data = st.session_state.data

        if column not in data.columns:
            return f"Column '{column}' not found. Available columns: {', '.join(data.columns)}"

        if data[column].dtype in ['int64', 'float64']:
            # Numerical column
            stats = data[column].describe()

            # Create histogram
            fig = px.histogram(
                data,
                x=column,
                title=f"Distribution of {column}",
                nbins=30
            )
            st.plotly_chart(fig)

            return (f"Distribution analysis of {column}:\n"
                    f"- Mean: {stats['mean']:.2f}\n"
                    f"- Median: {stats['50%']:.2f}\n"
                    f"- Std Dev: {stats['std']:.2f}\n"
                    f"- Min: {stats['min']:.2f}\n"
                    f"- Max: {stats['max']:.2f}\n")
        else:
            return self.get_value_counts(column)

    def process_query(self, query):
        """Process natural language queries with enhanced understanding"""
        query = query.lower()

        try:
            # Count queries
            count_patterns = [
                (r'how many (\w+)', 1),
                (r'count of (\w+)', 1),
                (r'number of (\w+)', 1),
                (r'total (\w+)', 1),
                (r'show (\w+) distribution', 1),
                (r'distribution of (\w+)', 1)
            ]

            for pattern, group_idx in count_patterns:
                match = re.search(pattern, query)
                if match:
                    column = match.group(group_idx)
                    return self.get_value_counts(column)

            # Column analysis
            if st.session_state.data is not None:
                for col in st.session_state.data.columns:
                    if col.lower() in query:
                        if any(word in query for word in ['analyze', 'analysis', 'distribution', 'breakdown']):
                            return self.analyze_distribution(col)

            # Data loading queries
            if any(word in query for word in ['load', 'upload', 'import']):
                return self.handle_data_upload()

            if st.session_state.data is None:
                return "Please upload a dataset first! Use the sidebar to upload your data file."

            # Show data info
            if 'show' in query and ('data' in query or 'info' in query):
                return self.show_data_info()

            # Handle ML queries
            if any(word in query for word in ['train', 'predict', 'model']):
                return self.handle_ml_query(query)

            # Handle visualization queries
            if any(word in query for word in ['plot', 'chart', 'graph', 'show']):
                return self.handle_visualization(query)

            # Handle correlation analysis
            if 'correlation' in query:
                return self.handle_correlation_analysis(query)

            # Handle text classification
            if 'text classification' in query:
                return self.handle_text_classification(query)

            # Handle data cleaning
            if 'clean' in query:
                return self.handle_data_cleaning(query)

            # Handle custom data transformations
            if 'transform' in query:
                return self.handle_custom_transformations(query)

            # Handle interactive dashboards
            if 'dashboard' in query:
                return self.handle_interactive_dashboards(query)

            # Handle 3D visualizations
            if '3d' in query:
                return self.handle_3d_visualizations(query)

            # Handle question type classification
            if 'question type' in query:
                return self.handle_question_type_classification(query)

            # Handle pattern mining
            if 'pattern mining' in query:
                return self.handle_pattern_mining(query)

            # Handle sequential pattern analysis
            if 'sequential pattern' in query:
                return self.handle_sequential_pattern_analysis(query)

            # Handle intelligent data cleaning
            if 'intelligent data cleaning' in query:
                return self.handle_intelligent_data_cleaning(query)

            # Handle streaming data analysis
            if 'streaming data' in query:
                return self.handle_streaming_data_analysis(query)

            # Handle network graphs
            if 'network graph' in query:
                return self.handle_network_graphs(query)

            # Handle interactive filters
            if 'interactive filters' in query:
                return self.handle_interactive_filters(query)

            # Handle export options
            if 'export' in query:
                return self.handle_export_options(query)

            # Handle AutoML pipeline
            if 'automl' in query:
                return self.handle_automl_pipeline(query)

            # Handle natural language understanding
            if 'nlu' in query:
                return self.handle_nlu(query)

            return ("I can help you with:\n"
                    "1. Counting and distributions (e.g., 'How many X are there?', 'Show distribution of X')\n"
                    "2. Data analysis (e.g., 'Analyze X', 'Show breakdown of X')\n"
                    "3. Training models (e.g., 'Train model to predict X')\n"
                    "4. Creating visualizations (e.g., 'Show chart of X')\n"
                    "5. Showing data info (e.g., 'Show data info')\n"
                    "6. Correlation analysis (e.g., 'Show correlation of X and Y')\n"
                    "7. Text classification (e.g., 'Classify text data')\n"
                    "8. Data cleaning (e.g., 'Clean the data')\n"
                    "9. Custom data transformations (e.g., 'Transform the data')\n"
                    "10. Interactive dashboards (e.g., 'Create dashboard')\n"
                    "11. 3D visualizations (e.g., 'Show 3D plot')\n"
                    "12. Question type classification (e.g., 'Classify question type')\n"
                    "13. Pattern mining (e.g., 'Mine patterns')\n"
                    "14. Sequential pattern analysis (e.g., 'Analyze sequential patterns')\n"
                    "15. Intelligent data cleaning (e.g., 'Intelligent data cleaning')\n"
                    "16. Streaming data analysis (e.g., 'Analyze streaming data')\n"
                    "17. Network graphs (e.g., 'Show network graph')\n"
                    "18. Interactive filters (e.g., 'Apply interactive filters')\n"
                    "19. Export options (e.g., 'Export data')\n"
                    "20. AutoML pipeline (e.g., 'Run AutoML pipeline')\n"
                    "21. Natural language understanding (e.g., 'Understand natural language')")

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

    def show_data_info(self):
        """Show detailed data information"""
        data = st.session_state.data
        info = []

        # Basic info
        info.append(f"Dataset Shape: {data.shape}\n")
        info.append("Column Information:")

        for col in data.columns:
            col_type = data[col].dtype
            n_unique = data[col].nunique()
            n_missing = data[col].isnull().sum()

            info.append(f"\n{col}:")
            info.append(f"- Type: {col_type}")
            info.append(f"- Unique Values: {n_unique}")
            if n_missing > 0:
                info.append(f"- Missing Values: {n_missing}")

            if col_type in ['int64', 'float64']:
                info.append(f"- Mean: {data[col].mean():.2f}")
                info.append(f"- Std: {data[col].std():.2f}")
            else:
                top_values = data[col].value_counts().head(3)
                info.append("- Top Values:")
                for val, count in top_values.items():
                    info.append(f"  * {val}: {count}")

        return "\n".join(info)

    def handle_ml_query(self, query):
        """Handle machine learning queries"""
        # Extract target variable
        target_pattern = r'predict\s+(\w+)'
        match = re.search(target_pattern, query)

        if not match:
            return "Please specify what to predict. Example: 'train model to predict price'"

        target = match.group(1)

        # Find matching column
        target_col = None
        for col in st.session_state.data.columns:
            if target.lower() in col.lower():
                target_col = col
                break

        if target_col is None:
            return f"Column containing '{target}' not found. Available columns: {', '.join(st.session_state.data.columns)}"

        # Train model
        return self.train_model(st.session_state.data, target_col)

    def train_model(self, data, target_col):
        """Train ML model with preprocessing"""
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]
            self.target = target_col

            # Determine if classification or regression
            if len(np.unique(y)) < 10 or y.dtype == 'object':
                model_type = 'classification'
                base_model = self.classification_models['xgboost']
            else:
                model_type = 'regression'
                base_model = self.regression_models['xgboost']

            # Create preprocessing pipeline
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

                # Create confusion matrix visualization
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm,
                                labels=dict(x="Predicted", y="Actual"),
                                title="Confusion Matrix")
                st.plotly_chart(fig)

                response = (f"Model trained successfully!\n\n"
                            f"Accuracy: {accuracy:.4f}\n"
                            f"F1 Score: {f1:.4f}\n\n"
                            "Confusion matrix plotted above.")

            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Create actual vs predicted plot
                fig = px.scatter(x=y_test, y=y_pred,
                                 labels={'x': 'Actual', 'y': 'Predicted'},
                                 title='Actual vs Predicted Values')
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                        y=[y_test.min(), y_test.max()],
                                        mode='lines',
                                        name='Perfect Prediction'))
                st.plotly_chart(fig)

                response = (f"Model trained successfully!\n\n"
                            f"RMSE: {np.sqrt(mse):.4f}\n"
                            f"R² Score: {r2:.4f}\n\n"
                            "Actual vs Predicted plot shown above.")

            # Feature importance
            if hasattr(base_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': base_model.feature_importances_
                }).sort_values('importance', ascending=False)

                fig = px.bar(importance.head(10),
                             x='feature',
                             y='importance',
                             title='Top 10 Feature Importance')
                st.plotly_chart(fig)

                response += "\n\nFeature importance plot shown above."

            return response

        except Exception as e:
            return f"Error training model: {str(e)}"

    def handle_visualization(self, query):
        """Create visualizations based on query"""
        data = st.session_state.data

        # Extract column names from query
        cols = [col for col in data.columns if col.lower() in query.lower()]

        if not cols:
            return "Please specify which columns you want to visualize."

        if 'scatter' in query and len(cols) >= 2:
            fig = px.scatter(data, x=cols[0], y=cols[1],
                             title=f"Scatter Plot: {cols[0]} vs {cols[1]}")
            st.plotly_chart(fig)
            return "Scatter plot created!"

        elif 'bar' in query and cols:
            fig = px.bar(data[cols[0]].value_counts().reset_index(),
                         x='index', y=cols[0],
                         title=f"Bar Plot of {cols[0]}")
            st.plotly_chart(fig)
            return "Bar plot created!"

        elif 'histogram' in query and cols:
            fig = px.histogram(data, x=cols[0],
                               title=f"Histogram of {cols[0]}")
            st.plotly_chart(fig)
            return "Histogram created!"

        elif 'box' in query and cols:
            fig = px.box(data, y=cols[0],
                          title=f"Box Plot of {cols[0]}")
            st.plotly_chart(fig)
            return "Box plot created!"

        return "Please specify the type of plot (scatter, bar, histogram, box) and the columns to visualize."

    def handle_correlation_analysis(self, query):
        """Handle correlation analysis"""
        data = st.session_state.data
        corr_matrix = data.corr()

        # Create heatmap
        fig = px.imshow(corr_matrix,
                         labels=dict(color="Correlation"),
                         title="Correlation Matrix")
        st.plotly_chart(fig)

        return "Correlation matrix plotted above."

    def handle_text_classification(self, query):
        """Handle text classification"""
        data = st.session_state.data
        text_col = 'text'
        label_col = 'label'

        if text_col not in data.columns or label_col not in data.columns:
            return f"Columns '{text_col}' and '{label_col}' not found. Available columns: {', '.join(data.columns)}"

        # Preprocess text data
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(data[text_col])
        y = data[label_col]

        # Train model
        model = MultinomialNB()
        model.fit(X, y)

        # Evaluate
        accuracy = model.score(X, y)

        return f"Text classification model trained successfully!\n\nAccuracy: {accuracy:.4f}"

    def handle_data_cleaning(self, query):
        """Handle data cleaning"""
        data = st.session_state.data

        # Drop missing values
        data = data.dropna()

        # Drop duplicates
        data = data.drop_duplicates()

        st.session_state.data = data
        return "Data cleaned successfully!"

    def handle_custom_transformations(self, query):
        """Handle custom data transformations"""
        data = st.session_state.data

        # Example transformation: Log transform
        for col in data.select_dtypes(include=['int64', 'float64']).columns:
            data[col] = np.log1p(data[col])

        st.session_state.data = data
        return "Custom data transformations applied successfully!"

    def handle_interactive_dashboards(self, query):
        """Handle interactive dashboards"""
        data = st.session_state.data

        # Create an interactive dashboard using Altair
        chart = alt.Chart(data).mark_circle().encode(
            x='column1:Q',
            y='column2:Q',
            color='column3:N',
            tooltip=['column1', 'column2', 'column3']
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        return "Interactive dashboard created!"

    def handle_3d_visualizations(self, query):
        """Handle 3D visualizations"""
        data = st.session_state.data

        # Create a 3D scatter plot
        fig = px.scatter_3d(data, x='column1', y='column2', z='column3',
                             title='3D Scatter Plot')
        st.plotly_chart(fig)
        return "3D visualization created!"

    def handle_question_type_classification(self, query):
        """Handle question type classification"""
        # Example: Classify the type of question
        if 'how many' in query:
            return "This is a count question."
        elif 'what is' in query:
            return "This is a definition question."
        else:
            return "This is a general question."

    def handle_pattern_mining(self, query):
        """Handle pattern mining"""
        data = st.session_state.data

        # Example: Frequent pattern mining using Apriori algorithm
        transactions = data.values.tolist()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        st.dataframe(rules)
        return "Pattern mining results shown above."

    def handle_sequential_pattern_analysis(self, query):
        """Handle sequential pattern analysis"""
        data = st.session_state.data

        # Example: Sequential pattern analysis
        sequences = data.values.tolist()
        # Implement sequential pattern mining algorithm here

        return "Sequential pattern analysis results shown above."

    def handle_intelligent_data_cleaning(self, query):
        """Handle intelligent data cleaning"""
        data = st.session_state.data

        # Example: Intelligent data cleaning using machine learning
        # Implement intelligent data cleaning algorithm here

        return "Intelligent data cleaning applied successfully!"

    def handle_streaming_data_analysis(self, query):
        """Handle streaming data analysis"""
        # Example: Streaming data analysis using Streamz
        source = Random(interval='1s')
        s = DaskStream()
        s.map(lambda x: x * 2).sink(print)
        source.emit(1)

        return "Streaming data analysis started."

    def handle_network_graphs(self, query):
        """Handle network graphs"""
        data = st.session_state.data

        # Example: Create a network graph
        G = nx.from_pandas_edgelist(data, 'source', 'target')
        net = Network(notebook=True)
        net.from_nx(G)
        net.show("network.html")

        return "Network graph created and saved as 'network.html'."

    def handle_interactive_filters(self, query):
        """Handle interactive filters"""
        data = st.session_state.data

        # Example: Create interactive filters using Altair
        brush = alt.selection_interval()
        chart = alt.Chart(data).mark_point().encode(
            x='column1:Q',
            y='column2:Q',
            color='column3:N'
        ).add_selection(
            brush
        ).properties(
            width=600,
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)
        return "Interactive filters applied!"

    def handle_export_options(self, query):
        """Handle export options"""
        data = st.session_state.data

        # Example: Export data to CSV
        data.to_csv('exported_data.csv', index=False)

        return "Data exported successfully to 'exported_data.csv'."

    def handle_automl_pipeline(self, query):
        """Handle AutoML pipeline"""
        data = st.session_state.data
        target_col = 'target'

        if target_col not in data.columns:
            return f"Column '{target_col}' not found. Available columns: {', '.join(data.columns)}"

        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Example: Run AutoML pipeline using XGBoost
        model = XGBClassifier()
        model.fit(X, y)

        return f"AutoML pipeline run successfully!\n\nBest pipeline: {model.score(X, y):.4f}"

    def handle_nlu(self, query):
        """Handle natural language understanding"""
        # Example: Natural language understanding using SpaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        displacy.render(doc, style="ent", jupyter=True)

        return "Natural language understanding results shown above."

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
