import json 
import random
import numpy as np

from tensorflow import keras
from colorama import Fore, Style, Back
from sklearn.preprocessing import LabelEncoder

def chat(model, data, tokenizer, label_encoder):
    print(f"{Fore.YELLOW}Start messaging with the bot (type quit to stop)!{Style.RESET_ALL}")

    while True:
        print(f"{Fore.LIGHTBLUE_EX}User:{Style.RESET_ALL}", end=" ")

        user_input = input().lower()
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]), truncating="post", maxlen=20))
        tag = label_encoder.inverse_transform([np.argmax(result)])

        for i in data["intents"]:
            if i["tag"] == tag:
                print(f"{Fore.GREEN}JAIDEN:{Style.RESET_ALL}", np.random.choice(i["responses"]))

                if i["tag"] == "goodbye": 
                    return