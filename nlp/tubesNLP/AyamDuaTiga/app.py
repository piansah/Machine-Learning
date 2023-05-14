from flask import Flask, render_template, request

# Inisialisasi Flask: Ini adalah bagian untuk menginisialisasi aplikasi Flask dengan memanggil objek Flask.
app = Flask(__name__)
app.static_folder = 'static'

# Import Library
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np 
from keras.models import load_model
model = load_model('Model4/models.h5')
import json
import random
intents = json.loads(open('model/dataset.json').read())
words = pickle.load(open('Model4/texts.pkl','rb'))
classes = pickle.load(open('Model4/labels.pkl','rb'))

# Clean Up Sentence: Ini adalah fungsi untuk membersihkan kalimat yang diterima dari pengguna, melakukan tokenisasi pada kalimat, dan melakukan lemmatization pada kata-kata.
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
# Bow (Bag of Words): Ini adalah fungsi untuk membuat bag of words dari kalimat yang diterima dari pengguna.
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Predict Class: Ini adalah fungsi untuk memprediksi inten dari kalimat yang diterima dari pengguna dengan menggunakan model machine learning.
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get Response: Ini adalah fungsi untuk mengambil jawaban dari dataset intents yang sesuai dengan inten yang diprediksi.
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# Chatbot Response: Ini adalah fungsi untuk memberikan jawaban dari chatbot dengan menggabungkan hasil dari fungsi predict_class dan getResponse.
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Route: Ini adalah bagian untuk mendefinisikan rute pada aplikasi Flask. Route home akan mengembalikan halaman HTML, sedangkan route get akan menerima pesan dari pengguna dan memberikan jawaban dari chatbot.
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

# Main Function: Ini adalah bagian yang akan dijalankan pertama kali saat aplikasi dijalankan, yaitu menjalankan server Flask.
if __name__ == "__main__":
    app.run()