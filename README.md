# CropAid - Plant Disease Detection Web App

CropAid is a simple web application that allows users to upload images of plant leaves and detect possible diseases using a machine learning model. The application also provides information about the detected disease and prevention tips using Google's Gemini API.

## Features

- User-friendly web interface
- Upload plant leaf images for disease detection
- Get predictions with confidence scores
- Access detailed disease information and prevention tips powered by Gemini AI
- Responsive design for both desktop and mobile devices

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Trained Keras model file (`crop_aid_plant_disease_recog_model.keras`)
- Gemini API key

### Installation

1. Clone or download this repository

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Add your Gemini API key in the `app.py` file:
   ```python
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```

4. Place your trained model file in the project root:
   ```
   crop_aid_plant_disease_recog_model.keras
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

```

## Customization

### Changing Class Names
If your model has different class names, update the `class_names` list in `app.py` to match your model's output classes.

### UI Customization
You can easily modify the appearance by editing the CSS file in `static/css/style.css`.
