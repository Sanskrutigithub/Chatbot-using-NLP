from flask import Flask, render_template, request, jsonify
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Flask app setup
app = Flask(__name__)

# Load intents file
with open('intents.json') as f:
    intents = json.load(f)

# Preprocess text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

# Train bag-of-words model
def train_bot():
    patterns = []
    tags = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            processed = preprocess(pattern)
            patterns.append(processed)
            tags.append(intent['tag'])
    
    return patterns, tags

patterns, tags = train_bot()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

def classify(sentence):
    processed_input = preprocess(sentence)
    input_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(input_vector, X)
    
    max_index = similarity.argmax()
    max_similarity = similarity[0, max_index]
    
    if max_similarity > 0.4:  # Adjust threshold as needed
        return tags[max_index]
    else:
        return "default"

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return i['responses'][0]
    return "I'm not sure how to respond to that."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    intent = classify(user_message)
    response = get_response(intent)
    return jsonify({'message': response, 'intent': intent})

if __name__ == '__main__':
    app.run(debug=True)
