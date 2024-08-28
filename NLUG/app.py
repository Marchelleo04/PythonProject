import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model
model = keras.models.load_model('chat-model')

# Load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

# Parameters
max_len = 20

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    if user_input:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, truncating='post', maxlen=max_len)

        result = model.predict(padded_sequence)
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                response = np.random.choice(i['responses'])
                return jsonify({"response": response})
    
    return jsonify({"response": "I didn't understand that. Can you please rephrase?"})

if __name__ == "__main__":
    app.run(debug=True)
