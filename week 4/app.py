from flask import Flask, render_template, request
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the iris dataset
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# Convert the data into a DataFrame
df = pd.DataFrame(data, columns=feature_names)
df['target'] = target

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(df.drop('target', axis=1), df['target'])

@app.route('/')
def index():
    return render_template('dashboard.html', data=df)

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    return render_template('dashboard.html', data=df, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
