from flask import Flask, render_template, request
import pickle, json, numpy as np

app = Flask(__name__)

model    = pickle.load(open("model.pkl",    "rb"))
scaler   = pickle.load(open("scaler.pkl",   "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
with open("metrics.json",  "r") as f: metrics  = json.load(f)
with open("features.json", "r") as f: FEATURES = json.load(f)

@app.route('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw = {
            'cgpa':               float(request.form['cgpa']),
            'internships':        int(request.form['internships']),
            'projects':           int(request.form['projects']),
            'workshops':          int(request.form['workshops']),
            'aptitude_score':     float(request.form['aptitude_score']),
            'soft_skills':        float(request.form['soft_skills']),
            'extracurricular':    request.form['extracurricular'],
            'placement_training': request.form['placement_training'],
            'ssc_marks':          float(request.form['ssc_marks']),
            'hsc_marks':          float(request.form['hsc_marks']),
        }

        encoded = {}
        for key, val in raw.items():
            if key in encoders:
                try:
                    encoded[key] = float(encoders[key].transform([val])[0])
                except ValueError:
                    encoded[key] = 0.0
            else:
                encoded[key] = float(val)

        arr    = np.array([[encoded[f] for f in FEATURES]])
        scaled = scaler.transform(arr)

        pred    = model.predict(scaled)[0]
        label   = encoders['status'].inverse_transform([pred])[0]
        prob    = model.predict_proba(scaled)[0]
        classes = encoders['status'].classes_
        pm      = {c: round(float(p)*100, 1) for c, p in zip(classes, prob)}

        placed_prob     = pm.get('Placed', 0)
        not_placed_prob = pm.get('Not Placed', 0)
        result_type     = 'placed' if label == 'Placed' else 'not_placed'

        return render_template('index.html', metrics=metrics,
            result='Student Will Get Placed' if result_type == 'placed' else 'Student Will NOT Get Placed',
            result_type=result_type,
            placed_prob=placed_prob,
            not_placed_prob=not_placed_prob)

    except Exception as e:
        import traceback; traceback.print_exc()
        return render_template('index.html', metrics=metrics,
            result=f'Error: {e}', result_type='error')

if __name__ == '__main__':
    app.run(debug=True)