from flask import Flask, render_template, request
import pickle
import urllib
from sklearn.feature_extraction.text import CountVectorizer
from newspaper import Article

clf_file = 'random_forest.pkl'
clf = pickle.load(open(clf_file, 'rb'))
tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))
#svd=pickle.load(open('svd_vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
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
	
    my_prediction = clf.predict_proba(vect)

    print(my_prediction)
	
    if my_prediction[0][0] >= 0.7:
        my_prediction = 0
    else:
        my_prediction = 1
	
    print(my_prediction)
	
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=False)
