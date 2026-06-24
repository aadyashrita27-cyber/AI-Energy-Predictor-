from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        hour = int(request.form["hour"])

        # Restrict input
        if hour < 1 or hour > 24:
            return render_template(
                "index.html",
                prediction="Please enter an hour between 1 and 24"
            )

        # Predict
        prediction = model.predict([[hour]])[0]

        # Load dataset for graph
        data = pd.read_csv("static/energy_data.csv")

        # Create graph
        plt.figure(figsize=(10, 5))

        # Original energy trend
        plt.plot(
            data["Hour"],
            data["Consumption"],
            marker="o",
            label="Actual Consumption"
        )

        # Highlight selected prediction
        plt.scatter(
           hour,
           prediction,
           s=200,
           color="red",
           marker="o",
           label="Predicted Value"
)

        plt.title("Energy Consumption vs Hour")
        plt.xlabel("Hour")
        plt.ylabel("Consumption (kWh)")
        plt.xticks(range(1, 25))
        plt.grid(True)
        plt.legend()

        plt.savefig("static/energy_graph.png")
        plt.close()

        prediction = round(prediction, 2)

    return render_template(
        "index.html",
        prediction=prediction
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


