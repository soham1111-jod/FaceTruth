from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import base64
import io
from werkzeug.utils import secure_filename
from model import MesoNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cpu")
model = MesoNet().to(device)

# Load model with error handling
try:
    model.load_state_dict(torch.load("mesonet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def predict(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            probs = outputs[0].cpu().detach().numpy()
            return probs
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error="No file selected")
        
        if file:
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                prob = predict(file_path)
                if prob is None:
                    return render_template('predict.html', error="Error processing image")
                
                prob = prob[0]
                prob_real = prob * 100
                prob_fake = (1 - prob) * 100
                prediction_label = 'Real' if prob > 0.5 else 'Fake'
                
                labels = ['Real', 'Fake']
                plt.figure(figsize=(6, 4), facecolor='#1F2937')
                ax = plt.gca()
                ax.set_facecolor('#1F2937')
                plt.bar(labels, [prob_real, prob_fake], color=['#10B981', '#EF4444'], width=0.5)
                plt.ylabel('Confidence (%)', color='white')
                plt.title('Detection Results', color='white', fontsize=14, fontweight='bold')
                plt.ylim(0, 100)
                plt.yticks([0, 20, 40, 60, 80, 100], color='white')
                plt.xticks(color='white')
                
                for i, v in enumerate([prob_real, prob_fake]):
                    plt.text(i, v + 2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                plt.close()  # Important: free memory
                buffer.seek(0)
                graph_string = base64.b64encode(buffer.getvalue()).decode()
                
                os.remove(file_path)
                
                return render_template('predict.html', 
                                       graph=graph_string, 
                                       prediction=prediction_label, 
                                       filename=filename,
                                       prob_real=f"{prob_real:.1f}",
                                       prob_fake=f"{prob_fake:.1f}")
            except Exception as e:
                print(f"Error: {e}")
                return render_template('predict.html', error="Error processing request")
    
    return render_template('predict.html')

@app.route('/ping')
def ping():
    return {'status': 'ok', 'message': 'FaceTruth API is running'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
