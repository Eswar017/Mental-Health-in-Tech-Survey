from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
#python app.py

app = Flask(__name__)

# Load model and scaler
model = joblib.load("mental_health_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Maps for encoding categorical inputs - match the label encoders used before
gender_map = {'male':0, 'female':1, 'other':2}
yes_no_map = {'No':0, 'Yes':1, "Don't know":2}
work_interfere_map = {'Never':0, 'Rarely':1, 'Sometimes':2, 'Often':3}
no_employees_map = {'1-5':0, '6-25':1, '26-100':2, '100-500':3, '500-1000':4, 'More than 1000':5}
mh_consequence_map = {'No':0, 'Yes':1, 'Maybe':2}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract inputs from form
            Age = float(request.form['Age'])
            Gender = gender_map[request.form['Gender']]
            self_employed = yes_no_map[request.form['self_employed']]
            family_history = yes_no_map[request.form['family_history']]
            work_interfere = work_interfere_map[request.form['work_interfere']]
            no_employees = no_employees_map[request.form['no_employees']]
            remote_work = yes_no_map[request.form['remote_work']]
            tech_company = yes_no_map[request.form['tech_company']]
            benefits = yes_no_map[request.form['benefits']]
            care_options = yes_no_map[request.form['care_options']]
            wellness_program = yes_no_map[request.form['wellness_program']]
            seek_help = yes_no_map[request.form['seek_help']]
            anonymity = yes_no_map[request.form['anonymity']]
            leave = yes_no_map[request.form['leave']]
            mental_health_consequence = mh_consequence_map[request.form['mental_health_consequence']]
            phys_health_consequence = mh_consequence_map[request.form['phys_health_consequence']]

            # Construct input dataframe
            input_df = pd.DataFrame([[Age, Gender, self_employed, family_history, work_interfere,
                                    no_employees, remote_work, tech_company, benefits, care_options,
                                    wellness_program, seek_help, anonymity, leave, mental_health_consequence,
                                    phys_health_consequence]],
                                    columns=['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
                                             'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
                                             'wellness_program', 'seek_help', 'anonymity', 'leave',
                                             'mental_health_consequence', 'phys_health_consequence'])
            # Scale input
            input_scaled = scaler.transform(input_df)

            # Predict
            pred = model.predict(input_scaled)[0]
            pred_proba = model.predict_proba(input_scaled)[0][pred]

            result = "Yes" if pred == 1 else "No"
            confidence = f"{pred_proba:.2f}"

            return render_template('index.html', result=result, confidence=confidence)
        except Exception as e:
            return render_template('index.html', error=str(e))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
