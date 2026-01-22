from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            pclass = int(request.form["Pclass"])
            sex = int(request.form["Sex"])
            age = float(request.form["Age"])
            fare = float(request.form["Fare"])
            embarked = int(request.form["Embarked"])

            input_data = np.array([[pclass, sex, age, fare, embarked]])
            input_scaled = scaler.transform(input_data)

            result = model.predict(input_scaled)[0]
            prediction = "Survived" if result == 1 else "Did Not Survive"

        except:
            prediction = "Invalid input"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
