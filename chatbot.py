import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function with error handling
def preprocess_text(text):
    try:
        if pd.isna(text):
            return ""
        tokens = word_tokenize(str(text).lower())
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized)
    except:
        return ""

# Load dataset from CSV
try:
    df = pd.read_csv('chatbot_dataset.csv')  # Replace with your CSV filename
    # Clean data - ensure required columns exist
    if 'Question' not in df.columns or 'Answer' not in df.columns:
        raise ValueError("CSV must contain 'Question' and 'Answer' columns")
    df = df.dropna(subset=['Question', 'Answer'])
    df['Question'] = df['Question'].astype(str)
    df['Answer'] = df['Answer'].astype(str)
    df['Processed'] = df['Question'].apply(preprocess_text)
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Fallback to sample data if CSV fails
    df = pd.DataFrame({
        'Question': ["What is your return policy?", "How do I contact support?"],
        'Answer': ["30 days money back guarantee", "Email us at support@company.com"]
    })
    df['Processed'] = df['Question'].apply(preprocess_text)

# Create model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Processed'])
model = NearestNeighbors(n_neighbors=1, algorithm='brute')
model.fit(X)

# Create Dash app
# Using a different theme for a fresh look, e.g., LUX or MINT
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX]) 
server = app.server

# Simple clean layout with improved styling
app.layout = dbc.Container([
    html.Div(className='header', children=[
        html.H1("IntelliChat Bot", className='title'),
        html.P("Your intelligent assistant is here to help!", className='subtitle')
    ]),
    
    dcc.Textarea(
        id='user-input',
        placeholder='Ask me anything...',
        className='input-box',
        rows=4 # Increased rows for better input visibility
    ),
    
    dbc.Button('Get Answer', 
               id='submit-button', 
               color='primary', # Keeping primary color, but style will be enhanced
               className='submit-btn'),
    
    html.Div(id='output-area', className='answer-box')
], className='main-container', fluid=True) # fluid=True makes it responsive

# Callback to handle questions
@app.callback(
    [Output('output-area', 'children'),
     Output('user-input', 'value')],
    [Input('submit-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def get_answer(n_clicks, question):
    if not n_clicks or not question:
        return dash.no_update, dash.no_update
    
    try:
        processed_q = preprocess_text(question)
        if not processed_q:
            return html.Div("Please enter a valid question.", className="bot-message error-message"), ""
            
        query_vec = vectorizer.transform([processed_q])
        dist, idx = model.kneighbors(query_vec)
        answer = df.iloc[idx[0][0]]['Answer']
        
        return [
            html.Div([
                html.P(f"You: {question}", className='user-question message'),
                html.P(f"Bot: {answer}", className='bot-answer message')
            ])
        ], ""
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return html.Div("Sorry, I couldn't find an answer to that question at the moment. Please try again later.", className="bot-message error-message"), ""

# Enhanced CSS styling - added directly for clarity and ease of modification
app.css.append_css({
    'external_url': (
        """
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
        }

        .main-container {
            background-color: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            max-width: 700px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 20px;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 20px;
        }

        .title {
            color: #2c3e50;
            font-weight: 700;
            font-size: 2.5em;
            margin-bottom: 5px;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: #7f8c8d;
            font-size: 1.1em;
            font-weight: 300;
        }

        .input-box {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            font-size: 1em;
            resize: vertical;
            min-height: 100px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .input-box::placeholder {
            color: #a0a0a0;
        }

        .input-box:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }

        .submit-btn {
            width: 100%;
            padding: 15px;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            background-image: linear-gradient(to right, #007bff, #0056b3);
            border: none;
            color: white;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 5px 15px rgba(0, 123, 255, 0.3);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(0, 123, 255, 0.4);
            background-image: linear-gradient(to right, #0056b3, #007bff);
        }

        .submit-btn:active {
            transform: translateY(0);
            box-shadow: 0 3px 10px rgba(0, 123, 255, 0.2);
        }

        .answer-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            box-shadow: inset 0 1px 5px rgba(0, 0, 0, 0.05);
            min-height: 80px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            transition: all 0.3s ease;
        }

        .message {
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 8px;
            line-height: 1.6;
        }

        .user-question {
            background-color: #e0f2fe; /* Light blue for user questions */
            color: #0056b3;
            align-self: flex-end; /* Align to the right */
            font-weight: 500;
            border-bottom-right-radius: 0;
        }

        .bot-answer {
            background-color: #e6ffe6; /* Light green for bot answers */
            color: #28a745;
            align-self: flex-start; /* Align to the left */
            font-weight: 400;
            border-bottom-left-radius: 0;
        }

        .error-message {
            background-color: #ffe6e6; /* Light red for error messages */
            color: #dc3545;
            font-weight: 500;
            text-align: center;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-container {
                padding: 25px;
                margin: 0 10px;
            }

            .title {
                font-size: 2em;
            }

            .subtitle {
                font-size: 1em;
            }

            .submit-btn, .input-box {
                font-size: 1em;
                padding: 12px;
            }
        }

        @media (max-width: 480px) {
            .main-container {
                padding: 15px;
                margin: 0 5px;
            }

            .title {
                font-size: 1.8em;
            }

            .subtitle {
                font-size: 0.9em;
            }
        }
        """
    )
})

if __name__ == '__main__':
    app.run(debug=True)