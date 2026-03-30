from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# model load
model = pickle.load(open("model.pkl", "rb"))

# home page
@app.route('/')
def home():
    return render_template("index.html")

# prediction
@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    result = model.predict([features])

    if result[0] == 1:
        output = "Accident Risk: YES"
    else:
        output = "Accident Risk: NO"

    return render_template("index.html", prediction_text=output)

# run app
if __name__ == "__main__":
    app.run(debug=True)