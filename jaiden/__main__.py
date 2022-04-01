import json
import pickle
import colorama

from core import chat
from tensorflow import keras

def main():
    model = keras.models.load_model('./models/chat_model')

    with open("./data/intents.json") as file: 
        data = json.load(file)
    with open('./models/tokenizer.pickle', 'rb') as handle: 
        tokenizer = pickle.load(handle)
    with open('./models/label_encoder.pickle', 'rb') as enc_file: 
        lbl_encoder = pickle.load(enc_file)

    colorama.init()
    chat(model, data, tokenizer, lbl_encoder)

    return 0

if __name__ == "__main__":
    main()