import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk import ngrams, FreqDist
from nltk.tokenize import word_tokenize
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier

#loading dataset
df = pd.read_csv('bbc-text.csv')
df.columns = ['category', 'text']

def preprocess(text):
    #remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    #convert to lowercase
    text = text.lower()
    #Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['text'] = df['text'].apply(preprocess)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size = 0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#evaluate accuracy
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#load first 50 news articles
text = ' '.join(df['text'][:50])
tokens = word_tokenize(text)
trigrams = list(ngrams(tokens, 3))
trigram_freq = FreqDist(trigrams)

#Generate 50 words
seed_word = random.choice(tokens)
output = [seed_word]
for _ in range(50):
    last_two = tuple(output[-2:])
    next_words = [trigram[2] for trigram in trigram_freq if trigram[:2] == last_two]
    if next_words:
        output.append(random.choice(next_words))
    else:
        break

print(' '.join(output))

#Generate tech news, 200 long
tech_words = df[df['category'] == 'tech']['text'].str.split().explode()
word_freq = FreqDist(tech_words)

def generate_text(length=200):
    text = []
    for _ in range(length):
        word = random.choices(list(word_freq.keys()), weights = list(word_freq.values()))[0]
        text.append(word)
    return ' '.join(text)

print(generate_text())

#Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names = df['category'].unique()))

#Train SGD model
sgd_model = SGDClassifier(loss='log_loss', max_iter=1000)
sgd_model.fit(X_train_vec, y_train)
y_pred_sgd = sgd_model.predict(X_test_vec)
accuracy_sgd = accuracy_score(y_test, y_pred_sgd)
print(f"SGD Accuracy: {accuracy_sgd:.2f}")
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")

