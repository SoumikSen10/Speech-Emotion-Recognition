from flask import Flask, request, jsonify
import numpy as np
import librosa
from keras.models import load_model
import os
import shutil
from werkzeug.utils import secure_filename
import logging
import atexit

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define essential configuration parameters
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
MODEL_PATH = 'speech_emotion_model.h5'

def cleanup_temp_folder():
    """Remove the temporary upload folder and its contents"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Successfully removed temporary folder: {UPLOAD_FOLDER}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary folder: {str(e)}")

# Register the cleanup function to run when the application exits
atexit.register(cleanup_temp_folder)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please ensure 'speech_emotion_model.h5' is in the working directory.")
model = load_model(MODEL_PATH)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path):
    """Extract MFCC features from audio file using the same parameters as training"""
    try:
        # Load audio with the same parameters used in training
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract MFCC features (using same parameters as training)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Reshape for model input
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        
        return mfcc
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    """Endpoint to predict emotion from audio file"""
    temp_file_path = None
    try:
        # Validate that a file was provided in the request
        if 'file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Save file temporarily with a secure filename
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_file_path)
        
        # Process the audio and make prediction
        features = extract_features(temp_file_path)
        prediction = model.predict(features, verbose=0)
        
        # Get predicted emotion
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
        emotion_index = np.argmax(prediction[0])
        predicted_emotion = emotions[emotion_index]
        
        return jsonify({"emotion": predicted_emotion})
                
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"})

def shutdown_cleanup():
    """Cleanup function to be called when the Flask app is shutting down"""
    cleanup_temp_folder()

# Register the cleanup function to run when Flask shuts down
atexit.register(shutdown_cleanup)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Ensure cleanup happens even if the app crashes
        cleanup_temp_folder()