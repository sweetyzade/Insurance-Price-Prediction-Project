from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3

app = Flask(__name__)

# Load the trained model
model = joblib.load('catboost_model.pkl')

# Function to preprocess user input
def preprocess_input(age, sex, bmi, children, smoker, region):
    # Convert sex to binary
    sex_binary = 1 if sex.lower() == 'male' else 0
    
    # Convert smoker to binary
    smoker_binary = 1 if smoker.lower() == 'yes' else 0
    
    # Convert region to one-hot encoded format
    region_mapping = {
        'northeast': [1, 0, 0, 0],
        'northwest': [0, 1, 0, 0],
        'southeast': [0, 0, 1, 0],
        'southwest': [0, 0, 0, 1]
    }
    region_encoded = region_mapping.get(region.lower())
    
    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_binary],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_binary],
        'region_0': [region_encoded[0]],
        'region_1': [region_encoded[1]],
        'region_2': [region_encoded[2]],
        'region_3': [region_encoded[3]]
    })
    
    return input_data

# Function to store user input and prediction into SQLite database
def store_data(age, sex, bmi, children, smoker, region, prediction):
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      age INTEGER, 
                      sex TEXT, 
                      bmi REAL, 
                      children INTEGER, 
                      smoker TEXT, 
                      region TEXT, 
                      prediction REAL)''')
    cursor.execute('''INSERT INTO predictions 
                      (age, sex, bmi, children, smoker, region, prediction) 
                      VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                      (age, sex, bmi, children, smoker, region, prediction))
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']
    
    # Preprocess input
    input_data = preprocess_input(age, sex, bmi, children, smoker, region)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Store data into SQLite database
    store_data(age, sex, bmi, children, smoker, region, prediction)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
