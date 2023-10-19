from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import urllib
from newspaper import Article
import pickle

# load the model from disk
filename = 'pac.pkl'
clf = pickle.load(open(filename, 'rb'))
tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))
svd=pickle.load(open('svd_vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	#if request.method == 'POST':
    #url = request.form['link']
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    data = [news]
	
    print(news)
    print(data)
	
    vect = tfidf.transform(data).toarray()

    print(vect)
	
    my_prediction = clf.predict(vect)
	
    print(my_prediction)
	
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=False)
