# Sentiment-Analysis
Binary classification of sentiment in a given text using count vectorizer and logistic regression machine learning model.

# Getting Started
For predicting sentiment of the text:

1. Run cell 14 of airline_sentiment_classification.ipynb file; with text to predict sentiment of that text

For testing API:
1. First intall fastapi and uvicorn using following commands:
- pip install fastapi
- pip install "uvicorn[standard]"
2. Run api.py file.
3. Go to the link provided after running sucessfully; in http address add "/docs"; click on routes, click try it out, enter data if needed and click execute. 

For Swagger Documentation file:
1. Change the server in the predict_sentiment.yaml file with the required user's swagger hub server
