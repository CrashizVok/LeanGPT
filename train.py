import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
chars = [] #Ez késöbb nem kell
words = [] #Ebben csak a szavak vannak
texts = [] #Ebben van az összes mondat

#Tokenizálás
with open("data.txt", "r", encoding="utf-8") as filen:
    for file in filen:
        i = file.strip()
        chars.append(i.strip().split())

for i in chars:
    for x in i:
        if x[-1] in [".", ",", ":", ";", "?","!","(",")"]:  
            word = x[:-1]  
            punctuation = x[-1]  
            words.append(word)  
            words.append(punctuation)  
        else:
            words.append(x) 
            
texts = words
words = list(dict.fromkeys(words)) 
if '<UNK>' not in words:
    words.append('<UNK>')
#####################################

word_to_index = {word: i for i, word in enumerate(words)}  # Szavak -> indexek
index_to_word = {i: word for word, i in word_to_index.items()}  # Indexek -> szavak


##################################


sequence_length = 40  
X = []
y = []

for i in range(len(words) - sequence_length):
    seq_in = words[i:i + sequence_length] 
    seq_out = words[i + sequence_length]  
    
    X.append([word_to_index[word] for word in seq_in])
    y.append(word_to_index[seq_out])

X = np.array(X)
y = np.array(y)

###################################
#   Modell létrehozása   #

model = tf.keras.Sequential([
    layers.Embedding(input_dim=len(words), output_dim=128, input_length=sequence_length),
    layers.Bidirectional(layers.LSTM(128)),
    layers.Dense(len(words), activation='softmax')
])


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=20, batch_size=32)

#model.save("text_generator_model.h5")

#########################################
#    Szöveg generálás vagy fasztudja #


def generate_text(model, seed_text, sequence_length, num_words=10):
    for _ in range(num_words):
        tokenized_input = [
            word_to_index.get(word, word_to_index['<UNK>'])
            for word in seed_text.split()
        ]
        tokenized_input = ([word_to_index['<UNK>']] * (sequence_length - len(tokenized_input)) + tokenized_input)[-sequence_length:]
        input_array = np.array(tokenized_input).reshape(1, sequence_length)
        predicted = model.predict(input_array, verbose=0).flatten()
        predicted_index = np.argmax(predicted)
        predicted_word = index_to_word[predicted_index]
        seed_text += " " + predicted_word
        if predicted_word == ".":
            break
    return seed_text

conversation_history = []

while True:
    print("SlimGPT: Helló! Kérdezz valamit vagy írj egy üzenetet.")
    seed_text = input("User: ")
    conversation_history.append(f"User: {seed_text}")
    context = " ".join(conversation_history[-5:])  # Utolsó 5 üzenet kontextusként
    response = generate_text(model, context, sequence_length, num_words=20)
    conversation_history.append(f"SlimGPT: {response}")
    print(f"SlimGPT: {response}")