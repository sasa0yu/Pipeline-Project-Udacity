  
"""
PREPROCESSING DATA
Disaster Response Pipeline Project
Udacity - Data Science Nanodegree
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function
    
    Arguments:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    Output:
        df -> Loaded dasa as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = pd.merge(categories, messages, how='inner', on=["id"])
    
    return df
    


def clean_data(df):
    """
    Clean Data function
    
    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame
    """
    
    # create a dataframe of the 36 individual category columns
    category_dicts = df["categories"].apply(lambda x : {temp.split("-")[0] : temp.split("-")[1] for temp in x.split(";")})
    categories = pd.DataFrame(category_dicts.tolist())
    df = pd.concat([df, categories], axis=1)
    
    #drop "categories"
    df = df.drop(columns = ["categories"])
    
    #drop duplicate
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save Data function
    
    Arguments:
        df -> Clean data Pandas DataFrame
        database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', con = engine, if_exists='replace', index=False)


def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """
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