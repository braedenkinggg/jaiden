import json
import pickle
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

model = Sequential()
lbl_encoder = LabelEncoder()
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")

labels = []
responses = []
training_labels = []
training_sentences = []

with open("./data/intents.json") as file:
    data = json.load(file)

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses.append(intent["responses"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

lbl_encoder.fit(training_labels)
tokenizer.fit_on_texts(training_sentences)

training_labels = lbl_encoder.transform(training_labels)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating="post", maxlen=20)

model.add(Embedding(1000, 16, input_length=20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(len(labels), activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(padded_sequences, np.array(training_labels), epochs=1000)
model.save("./models/chat_model")

with open("./models/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("./models/label_encoder.pickle", "wb") as enc_file:
    pickle.dump(lbl_encoder, enc_file, protocol=pickle.HIGHEST_PROTOCOL)