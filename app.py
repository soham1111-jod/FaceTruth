from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load ONNX model
try:
    ort_session = ort.InferenceSession("mesonet_model.onnx")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# ADDED: Custom static file route
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def predict(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1)).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: img_array}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        return ort_outputs[0][0]
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
                    os.remove(file_path)
                    return render_template('predict.html', error="Error processing image")
                
                prob = prob[0]
                prob_real = prob * 100
                prob_fake = (1 - prob) * 100
                prediction_label = 'Real' if prob > 0.5 else 'Fake'
                
                fig, ax = plt.subplots(figsize=(4, 3), facecolor='#1F2937')
                ax.set_facecolor('#1F2937')
                ax.bar(['Real', 'Fake'], [prob_real, prob_fake], 
                       color=['#10B981', '#EF4444'], width=0.5)
                ax.set_ylabel('Confidence (%)', color='white')
                ax.set_title('Detection Results', color='white', fontsize=12)
                ax.set_ylim(0, 100)
                ax.tick_params(colors='white')
                
                for i, v in enumerate([prob_real, prob_fake]):
                    ax.text(i, v + 2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
                
                for spine in ax.spines.values():
                    spine.set_visible(False)
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=72)
                plt.close(fig)
                buffer.seek(0)
                graph_string = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
                
                # Convert uploaded image to base64 for display
                # Detect image format from file extension
                file_ext = filename.lower().split('.')[-1]
                mime_type = 'image/jpeg' if file_ext in ['jpg', 'jpeg'] else f'image/{file_ext}'
                
                with open(file_path, 'rb') as img_file:
                    uploaded_img_base64 = base64.b64encode(img_file.read()).decode()
                
                os.remove(file_path)
                
                return render_template('predict.html', 
                                       graph=graph_string, 
                                       prediction=prediction_label, 
                                       uploaded_image=uploaded_img_base64,
                                       image_mime_type=mime_type,
                                       prob_real=f"{prob_real:.1f}",
                                       prob_fake=f"{prob_fake:.1f}")
                                       
            except Exception as e:
                print(f"Error: {e}")
                try:
                    os.remove(file_path)
                except:
                    pass
                return render_template('predict.html', error="Error processing request")
    
    return render_template('predict.html')

@app.route('/ping')
def ping():
    return {'status': 'ok', 'message': 'FaceTruth API is running'}, 200

@app.route('/health')
def health():
    return 'ok', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
