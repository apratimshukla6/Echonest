from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os


app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/back')
def back():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("train.csv",encoding='latin-1')
	df_data = df[["SentimentText","Sentiment"]]
	# Features and Labels
	df_x = df_data['SentimentText']
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus)
	# load the model from disk
	filename = 'finalized_model.sav'
	clf = pickle.load(open(filename, 'rb'))
	
	#Alternative Usage of Saved Model
	# ytb_model = open("naivebayes_spam_model.pkl","rb")
	# clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(host='0.0.0.0', port=port, debug=True)