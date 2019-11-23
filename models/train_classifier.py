
"""
TRAIN CLASSIFIER
Disaster Resoponse Project
Udacity - Data Science Nanodegree
How to run this script (Example)
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save ML model
"""


import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine, MetaData, Table, Column, String, Float, Integer
from sqlalchemy.sql import select
from sqlalchemy import func
import sqlite3
import os
import pickle
#os.getcwd()
#os.chdir('../')

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * FROM df", con = conn)
    X = df.message.values
    Y = df[list(set(df.columns.tolist()) - set(["id", "message", "original", "genre"]))]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, min_samples_split=2), n_jobs = -1))
    ])
    
    parameters = {
    #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    'features__text_pipeline__tfidf__use_idf': (True, False),
    #'clf__n_estimators': [50, 100, 200],
    #'clf__min_samples_split': [2, 3, 4],
    #'features__transformer_weights': (
    #    {'text_pipeline': 1, 'starting_verb': 0.5},
    #    {'text_pipeline': 0.5, 'starting_verb': 1},
    #    {'text_pipeline': 0.8, 'starting_verb': 1},
    #)
    }

    return GridSearchCV(pipeline, param_grid=parameters)

def evaluate_model(model, X_test, Y_test, category_names):
        """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """
    Y_pred = model.predict(X_test)
    for column, i in zip(Y_test.columns, range(36)):
        print(column)
        print(classification_report(Y_test[column].tolist(), Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    #StartingVerbExtractor.__module__ = "train_classifier"
    main()