from flask import Flask, jsonify, request, render_template

import random
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
import numpy as np
import pickle
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('app.html', name="")


@app.route('/train')
def train_data():
    lemmatizer = WordNetLemmatizer()

    words = []
    classes = []
    documents = []
    bigram_words = []

    ignore_words = [',', '|']

    intents = open('intents.json', encoding="utf-8").read()
    intents = json.loads(intents)

    print(intents)

    stopWords = set(stopwords.words('english'))

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            w = [word.replace("\\", "").replace("\"", "").replace(
                "\'", "").replace("?", "").replace("COVID-19", "").strip().lower() for word in w if word.isalnum()]
            w = [lemmatizer.lemmatize(
                word.lower()) for word in w]
            filtered_words = []
            for wrd in w:
                if wrd not in stopWords:
                    filtered_words.append(wrd)
            w = filtered_words
            bw = nltk.bigrams(w)
            bw = map(lambda x: ' '.join(x), list(bw))
            w.extend(list(bw))
            words.extend(w)
            documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    print(words)
    print(classes)
    print(documents)

    words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignore_words]

    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    print(len(documents), "documents")
    print(len(classes), "classes", classes)
    print(len(words), "unique lemmatized words", words)

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(
            word.lower())for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    print("Training data created")

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y),
                     epochs=300, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

    print("model created")
    return "Model created"


@app.route('/get_response', methods=['GET', 'POST'])
def create_response():
    lemmatizer = WordNetLemmatizer()
    model = load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json', encoding="utf-8").read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    def clean_up_sentence(sentence):
        stopWords = set(stopwords.words('english'))
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [word for word in sentence_words if word.isalnum()]
        sentence_words = [lemmatizer.lemmatize(
            word.lower()) for word in sentence_words]

        filtered_words = []
        for w in sentence_words:
            if w not in stopWords:
                filtered_words.append(w)

        bigrm = nltk.bigrams(filtered_words)

        result = map(lambda x: ' '.join(x), list(bigrm))

        filtered_words.extend(list(result))
        print(filtered_words)
        return filtered_words

    def bow(sentence, words, show_details=True):
        sentence_words = clean_up_sentence(sentence)
        bags = []
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag = [0]*len(words)
                    bag[i] = 1
                    bags.append(bag)
                    if show_details:
                        print('found in bag: %s' % w)
        return(np.array(bags))

    def predict_class(sentence):
        bags = bow(sentence, words)
        if(len(bags) == 0):
            return 0
        results = []
        for bag in bags:
            res = model.predict(np.array([bag]))[0]
            ERROR_THRESHOLD = 0.25
            PROBABILITY_THRESHOLD = 0
            result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
            result.sort(key=lambda x: x[1], reverse=True)
            results.append(result)

        print("results", results)
        return_list = []
        for result in results:
            for r in result:
                if (r[1] > PROBABILITY_THRESHOLD):
                    return_list.append(
                        {"intent": classes[r[0]], "probability": str(r[1])})
        print("return list", return_list)
        return return_list

    def getResponse(ints):
        list_of_intents = intents['intents']
        result = ints[-1]['intent']
        for i in list_of_intents:
            if(i['tag'] == result):
                return random.choice(i['responses'])

    def chatbot_response(msg):
        ints = predict_class(msg)
        if(ints == 0):
            sorry = ['Sorry, I don\'t understand what you mean.',
                     'Can you type in understandable words?']
            return(random.choice(sorry))
        res = getResponse(ints)
        return res

    print("You can start interact with the chatbot now.")

    # while True:
    user_input = request.form['keyword']
    user_input = user_input.lower().strip()

    if(user_input != ""):
        print("You: ============================================================================>>>", user_input)
        response = chatbot_response(user_input)
        print(
            "Bot: ============================================================================>>>", response)

    return response


if __name__ == '__main__':
    app.run(debug=True)
