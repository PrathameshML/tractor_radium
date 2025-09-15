from flask import Flask, request, render_template, redirect, url_for
import os
import sys
sys.path.append('model')
from predict import RadiumClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
classifier = RadiumClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            pred, conf = classifier.predict_video(filepath)
            if pred is None:
                result = 'No valid frames found.'
                label = 'Radium Absent'
                confidence = 0.0
            else:
                label = 'Radium Present' if pred == 1 else 'Radium Absent'
                confidence = f'{conf:.2f}'
            return render_template('result.html', filename=file.filename, result=label, confidence=confidence)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
