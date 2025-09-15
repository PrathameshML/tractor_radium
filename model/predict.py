import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os

MODEL_PATH = os.path.join('model', 'radium_classifier.pth')

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class RadiumClassifier:
    def __init__(self):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        self.model.eval()

    def predict_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = transforms.ToPILImage()(img)
        input_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        return pred, confidence

    def predict_video(self, video_path, frame_interval=10, end_frames=15):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_results = []
        frame_count = 0
        # Sample regular frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                pred, conf = self.predict_image(frame)
                frame_results.append((pred, conf))
            frame_count += 1
        # Sample more frames from the last few seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - end_frames))
        for i in range(end_frames):
            ret, frame = cap.read()
            if not ret:
                break
            pred, conf = self.predict_image(frame)
            frame_results.append((pred, conf))
        cap.release()
        if not frame_results:
            return None, 0.0
        # Prioritize radium present in last frames
        end_preds = [r[0] for r in frame_results[-end_frames:]]
        if end_preds.count(1) > 0:
            final_pred = 1
            avg_conf = np.mean([c for p, c in frame_results[-end_frames:] if p == 1])
        else:
            preds = [r[0] for r in frame_results]
            final_pred = max(set(preds), key=preds.count)
            avg_conf = np.mean([c for p, c in frame_results if p == final_pred])
        return final_pred, avg_conf
