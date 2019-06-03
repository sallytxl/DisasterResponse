# Disaster Response Pipeline Project

The project is to build a pipeline that can automatically classify the disaster related messages into different categories, which will facilitate the resuce team to provide the right aid. The "Disaster Response Messages" dataset from Figure Eight is used for training and testing of machine learning models. The text messages will be tokenized and critical words will be extracted for classification.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files in the repository
1. data
   -disaster_categories.csv: classification categories of messages
   -disaster_messages.csv: messages used for training and testing
   -process_data.py: ETL pipeline used to read, reorganize, clean data and save into an sqlite database
   -DisasterResponse.db: database used to store data after processed by ETL pipeline
   -YourDatabaseName.db: database used to store names of database
   
2. models
   -train_classifier.py: Python script to build and export a classification pipeline
   -classifier.pkl: trained classifier
  
3. app
   -templates: contain html file for the web application
   -run.py: flask file to run the web application
