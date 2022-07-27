import uvicorn
from fastapi import FastAPI
import joblib


# Loading sentiment classifer model and text to vector model
classifier = joblib.load(open("./model/sentiment_classifer.pkl","rb"))
vectorizer = joblib.load(open("./model/countvectorizer.pkl","rb"))

app = FastAPI()
#routes
@app.get('/')
async def index():
  return {"text":'Predict the sentiment of the text!'}


@app.post('/classify')
def classify(data : str):
    result = classifier.predict_proba(vectorizer.transform([data]).toarray())[0][1]
    # print(result)
    if result > 0.5:
        return {"Sentiment of text": 'Positive'}
    else:
        return {"Sentiment of text": 'Negative'}


if __name__ == '__main__':
    uvicorn.run(app)
    # uvicorn.run(app,host="127.0.0.1",port=8000)
