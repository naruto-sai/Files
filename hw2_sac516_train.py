import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import warnings


class logistic_models:
    """Class to create logistic regression models for text classification."""
    
    def __init__(self, df_x, df_y, hold_out=None, part_size=None, prep='norm', stp=True, tag='lemma', model='logistic',
                 repre='tfidf', ngrams=None):
        """
        Initialize the logistic_models class.
        
        Parameters:
        - df_x: DataFrame: The input features.
        - df_y: DataFrame: The target variable.
        - hold_out: DataFrame: Hold-out dataset for prediction (default=None).
        - part_size: float: Size of the test dataset (default=None).
        - prep: str: Preprocessing method (default='norm').
        - stp: bool: Whether to remove stopwords (default=True).
        - tag: str: Tagging method for preprocessing (default='lemma').
        - model: str: Type of model (default='logistic').
        - repre: str: Type of text representation (default='tfidf').
        - ngrams: tuple: N-grams range (default=None).
        """
        self.df_x = df_x
        self.df_y = df_y
        self.part_size = part_size
        self.stp = stp
        self.tag = tag
        self.model = model
        self.representation = repre
        self.ngrams = ngrams
        self.prep = prep
        self.hold_out = hold_out
        self.df_x_proc = self.df_x.text.apply(lambda x: self.preprocess(x))
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split()
        x_train_trans, x_test_trans = self.fitTransform(self.x_train, self.x_test)
        self.vectorizer = self.get_vectorizer()  
        y_pred = self.trainTest(x_train_trans, self.y_train, x_test_trans)
        self.accuracy, self.precision, self.recall, self.f1_score, self.cr = self.modelMetrics(y_pred, self.y_test)
        
    def train_test_split(self):
        """Split the data into training and testing sets."""
        X, y = self.df_x_proc, self.df_y
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.part_size, random_state=9, stratify=y) 
        print('x_train shape : ', x_train.shape, ' ', 'y train shape : ', y_train.shape, 
              '\nx_test shape  : ', x_test.shape, ' ', 'y test shape  : ', y_test.shape)
        return x_train, x_test, y_train, y_test
        
    def preprocess(self, text):
        """Preprocess the text data."""
        if self.prep:
            numb_iso = re.sub(r"(\d+\.\d+)", r" \1 ", text)
            clean_text = re.sub(r"[^\w\s]|_", "", numb_iso)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
        else:
            clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        if not self.stp:
            cleaned_text = ' '.join([j for j in clean_text.lower().split() if len(j) > 1])
        else:
            stop_w = stopwords.words('english')
            cleaned_text = ' '.join([j for j in clean_text.lower().split() if j not in stop_w if len(j) > 1])
        if self.tag == 'porter':
            porter = PorterStemmer()
            cleaned_tokens = [porter.stem(i) for i in cleaned_text.split()]
        elif self.tag == 'noPrep':
            return cleaned_text.split()
        else:
            lemmatizer = WordNetLemmatizer()
            cleaned_tokens = [lemmatizer.lemmatize(i) for i in cleaned_text.split()]
        return cleaned_tokens
    
    def fitTransform(self, train_data, test_data):
        """Fit and transform the text data."""
        if self.representation == 'bow':
            if self.ngrams is not None:
                self.vocab = CountVectorizer(ngram_range=self.ngrams)
            else:
                self.vocab = CountVectorizer()
            train_data= pd.Series(' '.join(i) for i in train_data)
            test_data = pd.Series(' '.join(i) for i in test_data)
            return self.vocab.fit_transform(train_data), self.vocab.transform(test_data)
        elif self.representation == 'tfidf':
            if self.ngrams:
                self.tfidf = TfidfVectorizer(ngram_range=self.ngrams)
            else:
                self.tfidf = TfidfVectorizer()
            train_data= pd.Series(' '.join(i) for i in train_data)
            test_data = pd.Series(' '.join(i) for i in test_data)
            return self.tfidf.fit_transform(train_data), self.tfidf.transform(test_data)
        elif self.representation == 'word2vec':
            model = Word2Vec(train_data, vector_size=100, window=5, min_count=1, workers=6)
            def vectorize(doc, model):
                vectors = [model.wv[token] for token in doc if token in model.wv.key_to_index]
                return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

            x_train = np.array([vectorize(doc, model) for doc in train_data])
            x_test = np.array([vectorize(doc, model) for doc in test_data])

            return x_train, x_test
    
    def get_vectorizer(self):
        """Get the vectorizer."""
        if self.representation == 'bow':
            return self.vocab
        elif self.representation == 'tfidf':
            return self.tfidf
    
    def trainTest(self, x_train, y_train, x_test):
        """Train and test the model."""
        warnings.filterwarnings('ignore')
        print(x_train.shape, y_train.shape, x_test.shape)
        if self.model == 'logistic':
            self.model = LogisticRegressionCV(cv=5, solver='liblinear', max_iter=1100)
            rfecv = RFECV(estimator=self.model, step=0.025, cv=StratifiedKFold(5), scoring='accuracy')
            rfecv.fit(x_train, y_train)
            x_train_selected = rfecv.transform(x_train)
            x_test_selected = rfecv.transform(x_test)
            self.model.fit(x_train_selected, y_train)
            if self.hold_out is not None:
                hold_out_proc = self.hold_out['text'].apply(lambda x: self.preprocess(x))
                hold_out_trans = self.fitTransform(self.x_train,hold_out_proc)[1] 
                pred_hold = self.model.predict(hold_out_trans)  
                self.hold_out['polite'] = pred_hold
                self.hold_out.to_csv(f'{self.prep}_{self.model}_{self.tag}.csv')
            return self.model.predict(x_test_selected)

    
    def top_features(self):
        """Get top features."""
        posit_coeff = self.model.coef_[0]

        top_indices = posit_coeff.argsort()[-5:][::-1]

        feature_names = self.vectorizer.get_feature_names()

        top_features = [feature_names[idx] for idx in top_indices]
        print(top_features)
    
    def modelMetrics(self, pred, actual):
        """Calculate model metrics."""
        accuracy = metrics.accuracy_score(actual, pred)
        precision = metrics.precision_score(actual, pred, average='binary')
        recall = metrics.recall_score(actual, pred, average='binary')
        f1_score = metrics.f1_score(actual, pred, average='binary')
        table = pd.DataFrame(metrics.classification_report(actual, pred, output_dict=True))
        print('\nAccuracy: ', accuracy)
        return accuracy, precision, recall, f1_score, table
    

class neural_net:
    """Class to create neural network models for text classification."""
    
    def __init__(self,df_x, df_y, hold_out=None, part_size=None, prep='norm', stp=True, tag='lemma',
                 repre='tfidf'):
        """
        Initialize the neural_net class.
        
        Parameters:
        - df_x: DataFrame: The input features.
        - df_y: DataFrame: The target variable.
        - hold_out: DataFrame: Hold-out dataset for prediction (default=None).
        - part_size: float: Size of the test dataset (default=None).
        - prep: str: Preprocessing method (default='norm').
        - stp: bool: Whether to remove stopwords (default=True).
        - tag: str: Tagging method for preprocessing (default='lemma').
        - repre: str: Type of text representation (default='tfidf').
        """
        self.df_x=df_x
        self.df_y = df_y
        self.hold_out=hold_out
        self.part_size = part_size
        self.stp = stp
        self.tag = tag
        self.representation = repre
        self.prep = prep
        self.df_x_proc = self.df_x.text.apply(lambda x: self.preproc(x))
        x_train, x_test, self.y_train, self.y_test = self.train_test_split()
        vocab, max_length, self.x_train, self.x_test = self.nn_prep(x_train, x_test)
        y_pred = self.nn_model(vocab, max_length)
        self.accuracy, self.precision, self.recall, self.f1_score, self.cr =self.modelMetrics(y_pred, self.y_test)
        
    def preproc(self, text):
        """Preprocess the text data."""
        if self.prep:
            numb_iso = re.sub(r"(\d+\.\d+)", r" \1 ", text)
            clean_text = re.sub(r"[^\w\s]|_", "", numb_iso)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
        else:
            clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        if not self.stp:
            cleaned_text = ' '.join([j for j in clean_text.lower().split() if len(j) > 1])
        else:
            stop_w = stopwords.words('english')
            cleaned_text = ' '.join([j for j in clean_text.lower().split() if j not in stop_w if len(j) > 1])
        if self.tag == 'porter':
            porter = PorterStemmer()
            cleaned_tokens = ' '.join([porter.stem(i) for i in cleaned_text])
        elif self.tag == 'noPrep':
            return cleaned_text
        else:
            lemmatizer = WordNetLemmatizer()
            cleaned_tokens = ' '.join([lemmatizer.lemmatize(i) for i in cleaned_text])
        return cleaned_tokens
    
    def train_test_split(self):
        """Split the data into training and testing sets."""
        X, y = self.df_x_proc, self.df_y
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.part_size, random_state=9, stratify=y) 
        print('x_train shape : ', x_train.shape, ' ', 'y train shape : ', y_train.shape, 
              '\nx_test shape  : ', x_test.shape, ' ', 'y test shape  : ', y_test.shape)
        return x_train, x_test, y_train, y_test
    
    def nn_prep(self, x_train, x_test):
        """Prepare data for neural network."""
        max_length = max(len(i) for i in x_train)
        self.tokenize = Tokenizer()
        self.tokenize.fit_on_texts(x_train)
        vocab = len(self.tokenize.word_index)
        train_data= self.tokenize.texts_to_sequences(x_train)
        x_train = pad_sequences(train_data, max_length, padding='post')
        test_data = self.tokenize.texts_to_sequences(x_test)
        x_test = pad_sequences(test_data, max_length, padding='post')
        
        return vocab, max_length, x_train, x_test
    
    def nn_model(self, vocab, max_length):
        """Define and train the neural network model."""
        model = Sequential()
        model.add(Embedding(input_dim=vocab+1, output_dim=100, input_length=max_length))
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(Bidirectional(LSTM(units=64)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, validation_split=0.1)

        # Prediction
        y_pred_prob = model.predict(self.x_test)

        y_pred = (y_pred_prob > 0.5).astype(int)
        if self.hold_out is not None:
            hold_out_proc = self.hold_out['text'].apply(lambda x: self.preproc(x))
            hold_out_trans = self.tokenize.texts_to_sequences(hold_out_proc)
            pred_hold = model.predict_classes(hold_out_trans)
            self.hold_out['polite'] = pred_hold
            self.hold_out.to_csv(f'{self.prep}_neural_{self.tag}.csv')

        return y_pred
    
    def modelMetrics(self, pred, actual):
        """Calculate model metrics."""
        accuracy = metrics.accuracy_score(actual, pred)
        precision = metrics.precision_score(actual, pred, average='binary')
        recall = metrics.recall_score(actual, pred, average='binary')
        f1_score = metrics.f1_score(actual, pred, average='binary')
        table = pd.DataFrame(metrics.classification_report(actual, pred, output_dict=True))
        print('\nAccuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1 Score: ', f1_score, '\n')
        return accuracy, precision, recall, f1_score, table