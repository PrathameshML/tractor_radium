# Tractor Trolley Radium Detection Web App

This project is a Flask-based web application for detecting the presence of a red radium sticker/board on the back of tractor trolleys from uploaded videos. The model is trained on images and predicts on video frames.

## Features
- Upload short videos (mp4) of tractor trolleys
- Detect if red radium is present at the back
- Binary and confidence score output
- Local storage of uploads and results

## Getting Started
1. Install Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Install PyTorch: `pip install torch torchvision`
4. Add your images to `data/radium_present` and `data/radium_absent`
5. Train the model: `python train_model.py`
6. Run the app: `python app.py`

## Dataset
- Images (jpg, jpeg, png) of tractor trolley backs
- Two classes: radium present, radium absent

## Tech Stack
- Flask (web framework)
- OpenCV (video frame extraction)
- PyTorch or TensorFlow (image classification)

## Usage
- Go to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser
- Upload a short mp4 video of a tractor trolley
- The app will display if red radium is present or absent, with confidence score

---
Project setup and prediction flow are complete.