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

# ML Models
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

            return ("I can help you with:\n"
                    "1. Counting and distributions (e.g., 'How many X are there?', 'Show distribution of X')\n"
                    "2. Data analysis (e.g., 'Analyze X', 'Show breakdown of X')\n"
                    "3. Training models (e.g., 'Train model to predict X')\n"
                    "4. Creating visualizations (e.g., 'Show chart of X')\n"
                    "5. Showing data info (e.g., 'Show data info')")

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
                base_model = self.classification_models['rf']
            else:
                model_type = 'regression'
                base_model = self.regression_models['rf']

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
