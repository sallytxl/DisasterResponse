import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
from joblib import dump, load
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Load the database and extract the messages, categories and category names.
    
    Keyword arguments:
    database_filepath -- location of the database
    
    Returns:
    X -- The dataset containing messages
    Y -- The labels
    Y.columns.values -- the label categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('test2', con = engine)
    df = df.dropna(subset=['related'])
    df = df[df['related'] > 0]
    print(df.shape[0])
    X = df['message']
    Y = df.iloc[:,4:40]
    return X, Y, Y.columns.values

def tokenize(text):
    """ Tokenize each message and remove useless words or symbols.
    
    Keyword arguments:
    text -- The message need to be tokenized
    
    Returns:
    token -- Tokens of a message
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    """ Build and optimize the model for message classficiation
    
    Keyword arguments:
    None
    
    Returns:
    cv -- the optimized model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    
    parameters =  { 'clf__estimator__n_estimators': [10,20], 
                    'clf__estimator__min_samples_split': [ 2,3] 
                  }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Test the model and print the classification metrics
    
    Keyword arguments:
    model -- the optimized classification model
    X_test -- the test data
    Y_test-- the test labels
    category_names -- the label names
    
    Returns:
    None
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print("Report for: ", col)
        target_name = ['class 0', 'class 1', 'class 2']
        print(classification_report(Y_test.loc[:, col], y_pred[:, i], target_names=target_name))

def save_model(model, model_filepath):
    """ Save the model to a file
    Keyword arguments:
    model -- the optimized classification model
    model_filepath -- the file path to save the model
    
    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(category_names)
        
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
    main()