from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

nltk.download('punkt', quiet=True)
warnings.filterwarnings('ignore')

# Grab the article
# article = Article('https://www.helpguide.org/articles/depression/depression-symptoms-and-warning-signs.htm')
article = Article('https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub/q-a-detail/coronavirus-disease-covid-19#:~:text=symptoms')

# put link inside the Article(link here) ^
article.download()
article.parse()
article.nlp()
corpus = article.text

text = corpus
sentence_list = nltk.sent_tokenize(text)  # listing the sentences

def greeting_response(text):
    text = text.lower()

    # set of respond
    bot_greetings = ['hello', 'hi', 'hey there', 'hey', 'wssup', 'hola']
    # user greeting
    user_greetings = ['hello', 'hi', 'hey there',
                      'hey', 'wassup', 'hola', 'greetings']

# ^nak tambah chatbot punya respond^

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)


def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))

    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:  # swap
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index


def bot_response(user_input):
    user_input = user_input.lower()
    sentence_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentence_list)
    # compare it to the rest of count metrics
    similarity_scores = cosine_similarity(cm[-1], cm)
    # reduce the dimention the similarity score
    similarity_scores_list = similarity_scores.flatten()
    # contain the indecies sorted the highest values in similarity score
    index = index_sort(similarity_scores_list)
    # contain only values that not itself
    index = index[1:]  # from 1 and onward
    response_flag = 0

    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:  # find the similarity
            bot_response = bot_response+' ' + sentence_list[index[i]]
            response_flag = 1
            j = j+1
        if j > 2:
            break
    if response_flag == 0:
        bot_response = bot_response+' '+"Sorry, I don't understand."

    sentence_list.remove(user_input)

    return bot_response


print('Doctor Bot: I am Doctor Bot or Doc Bot for short. I will answer your queries about about depression. If you want to exit type in the chat bye.')

exit_list = ['exit', 'see you later',
             'bye have a nice day', 'quit', 'break', 'bye']

while(True):
    user_input = input()
    if user_input.lower() in exit_list:
        print('\n'+'Doc Bot: bye bye'+'\n')
        break
    else:
        if greeting_response(user_input) != None:
            print('\n'+'Doc Bot: '+greeting_response(user_input)+'\n')
        else:
            print('\n'+'Doc Bot: '+bot_response(user_input)+'\n')
