import uvicorn
from fastapi import FastAPI
import joblib

classifier = joblib.load(open("sentiment_classifer.pkl","rb"))
vectorizer = joblib.load(open("countvectorizer.pkl","rb"))

app = FastAPI()

@app.get('/')
async def index():
  return {"text":'Sentiment Analysis on Tweets'}


@app.post('/classify')
def classify(data:str):
    # vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=20)
    result = classifier.predict_proba(vectorizer.transform([data]).toarray())[0][1]
    print(result)
    if result > 0.5:
        return {"Sentiment of text": 'Positive'}
    else:
        return {"Sentiment of text": 'Negative'}


if __name__ == '__main__':
    uvicorn.run(app)
    # uvicorn.run(app,host="127.0.0.1",port=8000)
