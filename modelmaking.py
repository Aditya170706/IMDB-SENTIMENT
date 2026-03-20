import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import string
df  = pd.read_csv("IMDB Dataset.csv")
df["sentiment"] = df["sentiment"].map({"positive":1 ,"negative":0})
df["review"] =df["review"].str.lower()  
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# download once
nltk.download('stopwords')



# load stopwords
stop_words = set(stopwords.words('english'))

def clean_text(sentence):
    # 1. Keep only alphabets and spaces
    sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
    
    # 2. Lowercase
    sentence = sentence.lower()
    
    # 3. Remove extra spaces
    words = sentence.split()
    
    # 4. Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

# apply on column
df['cleaned_review'] = df['review'].apply(clean_text)

vocab_size= 10000
from tensorflow.keras.preprocessing.text import Tokenizer


# create tokenizer
tokenizer = Tokenizer(num_words= vocab_size,oov_token="<OOV>")

# build vocabulary
tokenizer.fit_on_texts(df["cleaned_review"])

# convert text → sequences
sequences = tokenizer.texts_to_sequences(df["cleaned_review"])
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded = pad_sequences(sequences,maxlen=1440, padding='post')
max_len = df['cleaned_review'].apply(lambda x: len(x.split())).max()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional, 
                                      LSTM, Dense, Dropout)

model = Sequential([

    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    Dropout(0.2),

    # First BiLSTM — must have return_sequences=True
    # This passes the FULL sequence to the next layer
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),

    # Second BiLSTM — normal, no return_sequences needed
    Bidirectional(LSTM(32)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])
from tensorflow.keras.callbacks import EarlyStopping

# This STOPS training automatically when model stops improving
# So you don't have to babysit it!
early_stop = EarlyStopping(
    monitor='val_loss',    # watch the validation loss
    patience=3,            # stop if no improvement for 3 epochs
    restore_best_weights=True  # go back to the best version
)

model.compile(
    optimizer='adam',              # adam is fine, keep it
    loss='binary_crossentropy',    # standard for yes/no problems
    metrics=['accuracy']           # keep it simple for now
)

history = model.fit(
    padded, df['sentiment'],
    epochs=10,              # try 10, early stopping will handle the rest
    batch_size=64,          # process 64 reviews at a time
    validation_split=0.2,   # use 20% of data to check performance
    callbacks=[early_stop]
)