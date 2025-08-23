from flask import Flask, render_template
import csv
import os

app = Flask(__name__)

@app.route("/")
def index():
    data = []
    filename = "attendance.csv"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
