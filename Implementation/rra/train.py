import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tft
import re
import nltk
import spacy
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')

df = pd.read_csv('restaurant_reviews.csv')
df = df[df['Rating'] != 'Like']
df['Rating'] = df['Rating'].astype(float)
df['Target'] = df['Rating'].apply(lambda x: 0 if x > 3.0 else 1)

def clean_text(txt, stop=set(stopwords.words('english'))):
    txt = re.sub(r'\[nt]*', ' ', txt.lower())
    txt = re.sub(r'[^A-Za-z\s]', ' ', txt)
    txt = re.sub(r'[\s+]', ' ', txt)
    return " ".join([x for x in txt.split(' ') if x not in stop])

nlp = spacy.load('en_core_web_sm')
def lemmatize_review(txt):
    return ' '.join(word.lemma_ for word in nlp(txt))

df['clean_review'] = df['Review'].apply(clean_text).apply(lemmatize_review)

X, Y = df['clean_review'].to_numpy(), df['Target'].to_numpy()
dataset = tf.data.Dataset.from_tensor_slices((X, Y))

vect_layer = tf.keras.layers.TextVectorization(max_tokens=10000)
vect_layer.adapt(dataset.map(lambda x, y: x))

model = tf.keras.Sequential([
    vect_layer,
    tf.keras.layers.Embedding(len(vect_layer.get_vocabulary()), 128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation=None)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(X, Y, epochs=5, batch_size=32)

model.save('sentiment_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(vect_layer, f)

print("Model and tokenizer saved successfully!")

