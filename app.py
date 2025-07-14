from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import os

app = Flask(__name__)

# Load assets
model = load_model('chatbot/chatbot_model.h5')
words = pickle.load(open('chatbot/words.pkl', 'rb'))
classes = pickle.load(open('chatbot/classes.pkl', 'rb'))
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    app.run(debug=False)

