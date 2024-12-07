import os
import json
import nltk
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class NLPChatbotModel:
    def __init__(self, intents_file='intents.json'):
        """
        Initialize the NLP Chatbot Model
        
        Args:
            intents_file (str): Path to JSON file containing intents and patterns
        """
        # Lemmatizer for text normalization
        self.lemmatizer = WordNetLemmatizer()
        
        # Load intents from JSON file
        with open(intents_file, 'r') as file:
            self.intents = json.load(file)
        
        # Preprocessing variables
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_chars = ['?', '!', '.', ',']
        
        # Prepare training data
        self._prepare_training_data()
        
        # Create vectorizer for intent matching
        self.vectorizer = TfidfVectorizer()
        self._train_vectorizer()
        
        # Neural network for intent classification
        self.input_size = len(self.words)
        self.hidden_size = 128
        self.output_size = len(self.classes)
        self.model = self._build_neural_network()
    
    def _prepare_training_data(self):
        """
        Preprocess and organize training data
        """
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each pattern
                word_tokens = word_tokenize(pattern.lower())
                
                # Lemmatize and add to words
                lemmatized_words = [self.lemmatizer.lemmatize(w) for w in word_tokens 
                                    if w not in self.ignore_chars]
                
                self.words.extend(lemmatized_words)
                self.documents.append((lemmatized_words, intent['tag']))
                
                # Add to classes if not already present
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
    
    def _train_vectorizer(self):
        """
        Train TF-IDF vectorizer for intent matching
        """
        # Prepare patterns for vectorization
        patterns = [' '.join(doc[0]) for doc in self.documents]
        self.vectorizer.fit(patterns)
    
    def _build_neural_network(self):
        """
        Build a neural network for intent classification
        
        Returns:
            nn.Sequential: Neural network model
        """
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Softmax(dim=1)
        )
        return model
    
    def preprocess_input(self, sentence):
        """
        Preprocess user input
        
        Args:
            sentence (str): User input text
        
        Returns:
            list: Bag of words representation
        """
        # Tokenize and lemmatize
        word_tokens = word_tokenize(sentence.lower())
        lemmatized_words = [self.lemmatizer.lemmatize(w) for w in word_tokens 
                             if w not in self.ignore_chars]
        
        # Create bag of words
        bag = [1 if w in lemmatized_words else 0 for w in self.words]
        return bag
    
    def classify_intent(self, sentence):
        """
        Classify the intent of the user input
        
        Args:
            sentence (str): User input text
        
        Returns:
            str: Predicted intent
        """
        # Vectorize input
        input_vector = self.vectorizer.transform([' '.join(map(str, self.preprocess_input(sentence)))])
        
        # Compare with existing patterns
        similarities = cosine_similarity(input_vector, 
            self.vectorizer.transform([' '.join(map(str, doc[0])) for doc in self.documents]))
        
        # Get most similar intent
        most_similar_index = similarities.argmax()
        return self.documents[most_similar_index][1]
    
    def get_response(self, intent):
        """
        Generate a response based on the classified intent
        
        Args:
            intent (str): Classified intent
        
        Returns:
            str: Appropriate response
        """
        import random
        
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])
        return "I'm not sure how to respond to that."

# Flask Application Setup
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize the Chatbot Model
chatbot = NLPChatbotModel()

@app.route('/')
def home():
    """
    Render the home page of the chatbot application
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests
    """
    try:
        # Get user message from request
        user_message = request.json.get('message', '')
        
        # Classify intent
        intent = chatbot.classify_intent(user_message)
        
        # Generate response
        response = chatbot.get_response(intent)
        
        # Return response
        return jsonify({
            'message': response,
            'intent': intent
        })
    
    except Exception as e:
        return jsonify({
            'message': f'An error occurred: {str(e)}',
            'intent': 'error'
        }), 500

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NLP Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            border: 1px solid #ccc;
            height: 400px;
            overflow-y: scroll;
            padding: 10px;
        }
        #user-input {
            width: 70%;
            padding: 10px;
        }
        #send-button {
            padding: 10px;
        }
    </style>
</head>
<body>
    <h1>NLP Chatbot</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Type your message...">
    <button id="send-button">Send</button>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        function addMessage(message, sender) {
            const messageElement = document.createElement('div');
            messageElement.textContent = `${sender}: ${message}`;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;

            addMessage(message, 'You');
            userInput.value = '';

            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.message, 'Chatbot');
            })
            .catch(error => {
                console.error('Error:', error);
                addMessage('Sorry, something went wrong.', 'Chatbot');
            });
        }
    </script>
</body>
</html>
'''

# Intents Configuration
INTENTS_CONFIG = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Greetings"],
            "responses": [
                "Hello! How can I help you today?", 
                "Hi there! What can I do for you?", 
                "Greetings! Welcome to our chatbot."
            ]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "Goodbye", "See you later", "Exit"],
            "responses": [
                "Goodbye! Have a great day.", 
                "See you soon!", 
                "Take care!"
            ]
        },
        {
            "tag": "help",
            "patterns": ["Help", "What can you do", "Assistance"],
            "responses": [
                "I'm here to help! I can answer questions and provide information.",
                "I'm a helpful chatbot. Ask me anything!",
                "I'm ready to assist you with various queries."
            ]
        }
    ]
}

def create_project_structure():
    """
    Create project directory and necessary files
    """
    # Create project directory
    os.makedirs('nlp_chatbot_app', exist_ok=True)
    os.chdir('nlp_chatbot_app')
    
    # Create intents JSON file
    with open('intents.json', 'w') as f:
        json.dump(INTENTS_CONFIG, f, indent=4)
    
    # Create templates directory and index.html
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w') as f:
        f.write(HTML_TEMPLATE)
    
    # Create requirements file
    with open('requirements.txt', 'w') as f:
        f.write('''flask
flask-cors
nltk
numpy
scikit-learn
torch
''')
    
    # Create README
    with open('README.md', 'w') as f:
        f.write('''# NLP Chatbot Web Application

## Setup Instructions
1. Create a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK resources: `python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"`
4. Run the application: `python app.py`

## Features
- NLP-based intent classification
- Web interface for chatting
- Easily extendable intents
''')
    
    # Save the main application file
    with open('app.py', 'w') as f:
        f.write(open(__file__).read())

if __name__ == '__main__':
    # Uncomment the following line to create project structure
    create_project_structure()
    
    # Run the Flask app
    app.run(debug=True)