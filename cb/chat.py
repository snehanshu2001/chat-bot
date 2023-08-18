import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)

  # load trained model
model = keras.models.load_model('chat_model')

    # load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

def chat(inp):
  
    # parameters
    max_len = 20
    
    while True:
        

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                 return(np.random.choice(i['responses']))

    return "I do not understand"
            
        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = chat(sentence)
        print("Bot: "+resp)