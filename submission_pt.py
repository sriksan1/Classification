import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import *
from keras.layers import *
from keras.callbacks import *

train_df = pd.read_csv('train_2kmZucJ.csv')
X_train = train_df['tweet']
Y_train = train_df['label']
train_df.head()

def cleaner(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', ' ', text)
    # Remove special characters that aren't punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?\s]", " ", text)
    text = text.lower()
    tokens = text.split()
    return " ".join(tokens)
train_df['Clean Text'] = train_df['tweet'].apply(cleaner)
train_df['Clean Text'][1]

X_train,X_test,Y_train,Y_test = train_test_split(train_df['Clean Text'],train_df['label'],test_size=0.2,random_state=42,shuffle=True)
max_value_length = len(str(X_train.max()))
X_tokenizer = Tokenizer()
X_tokenizer.fit_on_texts(X_train)
X_tokenizer.word_index

len(X_tokenizer.word_index)
threshold = 3
cnt = 0
for key,value in X_tokenizer.word_index.items():
    if value >= threshold:
        cnt+=1
print(cnt)
X_tokenizer = Tokenizer(num_words=cnt,oov_token='unk')
X_tokenizer.fit_on_texts(X_train)
from keras.preprocessing.sequence import pad_sequences
#79 is the longest length sequence so set the padding to 100
max_len = 100
X_train_seq = X_tokenizer.texts_to_sequences(X_train)
X_test_seq = X_tokenizer.texts_to_sequences(X_test)
X_train_seq = pad_sequences(X_train_seq,padding='post',maxlen = max_len)
X_test_seq = pad_sequences(X_test_seq,padding='post',maxlen = max_len)
x_voc_size = X_tokenizer.num_words


model = Sequential()
model.add(Embedding(x_voc_size,50,input_shape=(max_len,),mask_zero=True))
#rnn
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy')
mc = ModelCheckpoint("weights.best.keras",monitor='val_loss',verbose=1,save_best_only=True,mode='min')
model.fit(X_train_seq,Y_train,batch_size=128,epochs=10,verbose=1,validation_data=(X_test_seq,Y_test),callbacks=[mc])
from sklearn import metrics
model.load_weights("weights.best.keras")
results_predicted = model.predict(X_test_seq)
predicted_class = (model.predict(X_test_seq) > 0.5).astype("int32")

Y_pred = pd.Series(predicted_class.flatten(), name="Predicted_Class")
print(metrics.classification_report(Y_test,Y_pred))
test_file = r'test_12QyDcx.csv'
df = pd.read_csv(test_file)
X = df.tweet
X_clean = X.apply(cleaner)
X_clean_seq = X_tokenizer.texts_to_sequences(X_clean)
X = pad_sequences(X_clean_seq,padding='post',maxlen = max_len)
results_example_file = (model.predict(X) > 0.5).astype('int')
df['Label'] = results_example_file
df.to_csv('Submission.csv')