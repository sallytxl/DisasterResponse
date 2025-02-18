import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load csv files which contains messages and categories information.
    
    Keyword arguments:
    messages_filepath -- location of message csv file
    categories_filepath -- location of categories csv file
    
    Returns:
    df -- Dataframe containing both message and category info
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath, dtype={'categories':'str'})
    df = messages.merge(categories, on='id')
    
    temp = pd.read_csv(categories_filepath, dtype={'categories':'str'})
    categories = temp['categories'].str.split(';', expand = True)
    
    row = categories.iloc[0]
    
    category_colnames = [w[:-2] for w in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    
    categories['id'] = temp['id']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort = False, join='outer', axis=1)
    
    return df

def clean_data(df):
    """ Remove duplicated rows.
    
    Keyword arguments:
    df -- original data set
    
    Returns:
    df -- data set with out duplicated rows
    """
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """ Save dataframe to a sqlite database.
    
    Keyword arguments:
    df -- dataframe to be saved
    database_filename -- destination database
    
    Returns:
    this function doesn't return anything
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('test2', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
