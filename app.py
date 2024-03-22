from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        review = request.form['review']

        model = joblib.load("naive_bayes.pkl")
        
        sentiment = model.predict([review])[0]
        
        sentiment_label = 'Positive' if sentiment == 1 else 'Negative'
        
        return render_template('predict.html', review=review, sentiment=sentiment_label)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
