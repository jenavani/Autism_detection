from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('Autism/models/autism_model.joblib')
label = joblib.load('Autism/models/label.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Handle form data
            data = {
                'a1_score': int(request.form['a1_score']),
                'a2_score': int(request.form['a2_score']),
                'a3_score': int(request.form['a3_score']),
                'a4_score': int(request.form['a4_score']),
                'a5_score': int(request.form['a5_score']),
                'a6_score': int(request.form['a6_score']),
                'a7_score': int(request.form['a7_score']),
                'a8_score': int(request.form['a8_score']),
                'a9_score': int(request.form['a9_score']),
                'a10_score': int(request.form['a10_score']),
                'age': int(request.form['age']),
                'gender': 0 if request.form['gender'] == 'male' else 1,
                'autism_for_immediate_family_members': 0 if request.form['autism_for_immediate_family_members'] == 'no' else 1,
                'Country': label.transform([request.form['Country']])[0],
                'used_app_before_for_test': 0 if request.form['used_app_before_for_test'] == 'no' else 1,
                'result': float(request.form['result']),
            }

            input_data = [
                data['a1_score'], data['a2_score'], data['a3_score'], data['a4_score'], data['a5_score'],
                data['a6_score'], data['a7_score'], data['a8_score'], data['a9_score'], data['a10_score'],
                data['age'], data['gender'], data['autism_for_immediate_family_members'],
                data['Country'], data['used_app_before_for_test'], data['result']
            ]

            prediction = model.predict([input_data])
            predict_result = 'Autism' if prediction[0] == 1 else 'Not Autism'
            return render_template('result.html', prediction=predict_result)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = [
            data['a1_score'], data['a2_score'], data['a3_score'], data['a4_score'], data['a5_score'],
            data['a6_score'], data['a7_score'], data['a8_score'], data['a9_score'], data['a10_score'],
            data['age'], data['gender'], data['autism_for_immediate_family_members'],
            data['Country'], data['used_app_before_for_test'], data['result']
        ]

        prediction = model.predict([input_data])
        predict_result = 'Autism' if prediction[0] == 1 else 'Not Autism'
        return predict_result
    except Exception as e:
        return jsonify({"Error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)