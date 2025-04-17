from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import tensorflow as tf
import os
import uuid
import json
from pathlib import Path
import google.generativeai as genai
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetB4
import re

print(f"TensorFlow version: {tf.__version__}")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# App configuration for file uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create temp model directory if it doesn't exist
os.makedirs('temp_model', exist_ok=True)

# List of available Gemini AI models
GEMINI_MODELS = [
    {"id": "models/gemini-1.5-flash", "name": "Gemini 1.5 Flash", "description": "Fast, versatile multimodal model"},
    {"id": "models/gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash-8B", "description": "Optimized for high-volume tasks"},
    {"id": "models/gemini-1.5-pro", "name": "Gemini 1.5 Pro", "description": "For complex reasoning tasks"},
    {"id": "models/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "description": "Enhanced performance with low latency"},
    {"id": "models/gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash-Lite", "description": "Cost-efficient model"},
    {"id": "models/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "description": "Google's most advanced AI model"}
]

# Define class names (disease categories)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load disease information from JSON file
try:
    with open("plant_disease.json", 'r') as file:
        plant_disease = json.load(file)
    print(f"Loaded {len(plant_disease)} disease entries from JSON")
except Exception as e:
    print(f"Error loading plant_disease.json: {e}")
    plant_disease = []

# Custom template filter for formatting prevention tips
@app.template_filter('format_prevention_tips')
def format_prevention_tips(text):
    """Convert numbered list in plain text to formatted HTML with tip-item classes and handle markdown formatting"""
    if not text:
        return ""
    
    # Convert markdown bold (** **) to HTML <strong> tags
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert markdown italic (* *) to HTML <em> tags
    text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', text)
    
    # Replace numbered list items with styled divs
    text = re.sub(r'1\.\s*', r'<div class="tip-item"><span class="tip-number">1</span> ', text)
    text = re.sub(r'2\.\s*', r'</div><div class="tip-item"><span class="tip-number">2</span> ', text)
    text = re.sub(r'3\.\s*', r'</div><div class="tip-item"><span class="tip-number">3</span> ', text)
    text = re.sub(r'4\.\s*', r'</div><div class="tip-item"><span class="tip-number">4</span> ', text)
    text = re.sub(r'5\.\s*', r'</div><div class="tip-item"><span class="tip-number">5</span> ', text)
    
    # Close the last div
    text += '</div>'
    
    return text

# Configure Gemini API with user's API key
def configure_genai_with_key(api_key, model_id=None):
    """Configure the Gemini API with the user's key."""
    try:
        genai.configure(api_key=api_key, transport="rest")
        if not model_id:
            model_id = GEMINI_MODELS[0]["id"]
        print(f"Using model: {model_id}")
        model = genai.GenerativeModel(model_id)
        response = model.generate_content("Hello")
        session['gemini_model_name'] = model_id
        return True, model, model_id
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return False, str(e), None

# Get detailed disease information from Gemini API
def get_disease_info(disease_name, api_key, model_id=None):
    try:
        if not model_id:
            model_id = session.get('gemini_model_name')
        
        # Configure Gemini API if not already done
        if 'gemini_model' not in session or not model_id:
            success, model_or_error, model_id = configure_genai_with_key(api_key, model_id)
            if not success:
                return "Could not retrieve information. Please check your API key."
            model = model_or_error
        else:
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel(model_id)
        
        # Create appropriate prompt based on whether the plant is healthy or diseased
        if disease_name.endswith('healthy'):
            plant_type = disease_name.split('___')[0]
            prompt = f"Generate a short paragraph (max 150 words) explaining that {plant_type} plants appear healthy, with tips for maintaining plant health."
        else:
            prompt = f"Generate a short paragraph (max 150 words) about {disease_name.replace('___', ' ')} disease in plants. Include symptoms, causes, and treatment options."
        
        # Generate content with Gemini API
        response = model.generate_content(prompt)
        response_text = response.text
        response_text = response_text.replace('&lt;', '<').replace('&gt;', '>')
        return response_text
    except Exception as e:
        print(f"Error getting disease info: {str(e)}")
        return f"Could not retrieve additional information: {str(e)}"

# Get prevention tips from Gemini API
def get_prevention_tips(disease_name, api_key, model_id=None):
    try:
        if not model_id:
            model_id = session.get('gemini_model_name')
        
        # Configure Gemini API if not already done
        if 'gemini_model' not in session or not model_id:
            success, model_or_error, model_id = configure_genai_with_key(api_key, model_id)
            if not success:
                return "Could not retrieve information. Please check your API key."
            model = model_or_error
        else:
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel(model_id)
        
        # Create appropriate prompt based on whether the plant is healthy or diseased
        if disease_name.endswith('healthy'):
            plant_type = disease_name.split('___')[0]
            prompt = f"List 5 best practices for maintaining the health of {plant_type} plants."
        else:
            prompt = f"List 5 prevention tips for {disease_name.replace('___', ' ')} disease in plants."
        
        # Generate content with Gemini API
        response = model.generate_content(prompt)
        response_text = response.text
        response_text = response_text.replace('&lt;', '<').replace('&gt;', '>')
        return response_text
    except Exception as e:
        print(f"Error getting prevention tips: {str(e)}")
        return f"Could not retrieve prevention tips: {str(e)}"

# Check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the trained model
def load_trained_model():
    """Load the trained model for plant disease classification"""
    try:
        print("Loading trained model...")
        model_path = 'crop_aid_plant_disease_recog_model.keras'
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at: {model_path}")
            return None, False
        
        # Simple loading approach to avoid compatibility issues
        try:
            # First try simple loading
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile model with simple settings
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded successfully")
            return model, True
            
        except Exception as e1:
            print(f"Standard loading failed: {e1}")
            print("Creating model architecture manually...")
            
            # Create the model architecture manually
            base_model = EfficientNetB4(
                input_shape=(160, 160, 3),
                include_top=False,
                weights=None  # Don't load ImageNet weights
            )
            
            inputs = tf.keras.Input(shape=(160, 160, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            print("Loading weights...")
            model.load_weights(model_path)
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Model created and weights loaded")
            return model, True
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# Process image for prediction
def process_image(image_path):
    """Process image for prediction"""
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(160, 160))
        img_array = img_to_array(img)
        
        # Use EfficientNet preprocessing (matches the model architecture)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

# Predict disease from image
def predict_disease(image_path, api_key=None, model_id=None, use_gemini=False):
    """Predict disease from image"""
    # Check if model is loaded
    if model is None:
        print("ERROR: Model is not loaded!")
        return {"error": "Model not loaded. Please check server logs."}
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file does not exist: {image_path}")
        return {"error": "Image file not found"}
    
    # Process image
    img_array = process_image(image_path)
    if img_array is None:
        return {"error": "Failed to process image"}
    
    # Make prediction
    try:
        # Get predictions
        predictions = model.predict(img_array, verbose=0)
        
        # Get predicted class
        prediction_index = np.argmax(predictions[0])
        disease_name = class_names[prediction_index]
        confidence = float(predictions[0][prediction_index])
        print(f"Prediction: {disease_name} with confidence {confidence:.6f}")
        
        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[-5:][::-1]
        print("\nTop 5 predictions:")
        for i, idx in enumerate(top5_indices):
            print(f"  {i+1}. {class_names[idx]}: {predictions[0][idx]:.6f}")
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    # Get disease info from JSON
    disease_info = None
    for disease in plant_disease:
        if disease["name"] == disease_name:
            disease_info = disease
            break
    
    # Get additional information if API key is provided
    gemini_info = ""
    prevention_tips = ""
    
    if api_key and use_gemini:
        gemini_info = get_disease_info(disease_name, api_key, model_id)
        prevention_tips = get_prevention_tips(disease_name, api_key, model_id)
    
    # Create result
    result = {
        'name': disease_name,
        'class': disease_name.replace('___', ' '),
        'confidence': f"{confidence * 100:.2f}%",
        'cause': disease_info["cause"] if disease_info else "Information not available",
        'cure': disease_info["cure"] if disease_info else "Information not available",
        'disease_info': gemini_info,
        'prevention_tips': prevention_tips,
        'top_predictions': [
            {'class': class_names[idx].replace('___', ' '), 
             'confidence': f"{predictions[0][idx] * 100:.2f}%"}
            for idx in top5_indices
        ]
    }
    
    return result

# Load the model
model, model_loaded = load_trained_model()

# Route for homepage
@app.route('/')
def home():
    if not model_loaded:
        flash("Model could not be loaded. Please contact administrator.", "danger")
    api_key = session.get('api_key', '')
    model_id = session.get('gemini_model_name', '')
    return render_template('index.html', api_key=api_key, models=GEMINI_MODELS, selected_model=model_id)

# Route for setting API key
@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    api_key = request.form.get('api_key', '')
    model_id = request.form.get('model_id', GEMINI_MODELS[0]['id'])
    
    if api_key:
        success, model_or_error, used_model_id = configure_genai_with_key(api_key, model_id)
        if success:
            session['api_key'] = api_key
            session['gemini_model'] = True
            session['gemini_model_name'] = used_model_id
            flash('API key set successfully!', 'success')
        else:
            flash(f'Invalid API key: {model_or_error}', 'danger')
    else:
        flash('Please enter an API key', 'warning')
    
    return redirect(url_for('home'))

# Route for clearing API key
@app.route('/clear_api_key', methods=['POST'])
def clear_api_key():
    session.pop('api_key', None)
    session.pop('gemini_model', None)
    flash('API key cleared', 'info')
    return redirect(url_for('home'))

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        flash("Model could not be loaded. Please contact administrator.", "danger")
        return redirect(url_for('home'))
    
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('home'))
    
    file = request.files['file']
    api_key = request.form.get('api_key', session.get('api_key', ''))
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        # Save file
        filename = str(uuid.uuid4()) + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if Gemini API should be used
        use_gemini = request.form.get('use_gemini') == 'on' and api_key
        model_id = session.get('gemini_model_name')
        
        # Get prediction
        result = predict_disease(filepath, api_key, model_id, use_gemini)
        
        # Render result
        if "error" in result:
            flash(f"Error: {result['error']}", "danger")
            return redirect(url_for('home'))
        
        return render_template('result.html', 
                              result=result, 
                              image_path=filename,
                              api_key=api_key,
                              use_gemini=use_gemini)
    else:
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
        return redirect(url_for('home'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)