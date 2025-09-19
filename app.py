

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import sys
sys.path.append('model')
from predict import RadiumClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
classifier = RadiumClassifier()

# Route to serve uploaded files (detected frames)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        filetype = request.form.get('filetype')
        if filetype == 'image':
            file = request.files.get('image')
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                import cv2
                img = cv2.imread(filepath)
                if img is None:
                    label = 'Invalid image file.'
                    confidence = 0.0
                else:
                    pred, conf = classifier.predict_image(img)
                    label = 'Radium Present' if pred == 1 else 'Radium Absent'
                    confidence = f'{conf:.2f}'
                return render_template('result.html', filename=file.filename, result=label, confidence=confidence)
        else:
            file = request.files.get('video')
            if file:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                pred, conf, frame_path = classifier.predict_video(
                    filepath,
                    save_frame_dir=app.config['UPLOAD_FOLDER'],
                    save_frame_prefix=os.path.splitext(file.filename)[0]
                )
                if pred is None:
                    label = 'No valid frames found.'
                    confidence = 0.0
                    frame_url = None
                else:
                    label = 'Radium Present' if pred == 1 else 'Radium Absent'
                    confidence = f'{conf:.2f}'
                    frame_url = None
                    if pred == 1 and frame_path:
                        frame_url = url_for('uploaded_file', filename=os.path.basename(frame_path))
                return render_template('result.html', filename=file.filename, result=label, confidence=confidence, detected_frame=frame_url)
    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
