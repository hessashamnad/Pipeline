import sys
import numpy as np
import pandas as pd
import re
import nltk
nltk.download(['punkt','wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sqlalchemy import create_engine
import sys
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,fbeta_score, make_scorer, accuracy_score

def load_data(database_filepath):
    """
    Load Data Function
    
    input:
         database name
    outputs:
        X: messages 
        y: rest of 
        category names
    """
   
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_disaster', engine)
    X = df['message']
    Y = df[df.columns[5:]]

    category_names = list(np.array(Y.columns))

    return X, Y, category_names


def tokenize(text) :
    '''
    Function for tokenizing string
    Args: Text string
    Returns: List of tokens
    '''
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    create pipeline
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())])
    
    
    parameters = {
        'clf__min_samples_split': [5,10, 15],
        'clf__n_estimators': [50, 100, 150]}

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs: model, X_test, y_test, category_names
    output: scores
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  i, cat in enumerate(Y_test.columns.values):
        print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,i], y_pred[:, i])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))
    
def save_model(model, model_filepath):
    """
    Pickled Fitted model
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
    main()