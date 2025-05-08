from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static')

# Load the model
model_path = 'Model.pickle'
model = None

def load_model():
    global model
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Load model at startup
load_model()

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        input_values = data['values']
        
        # Convert to numpy array and reshape
        input_array = np.array(input_values).astype(float).reshape(1, -1)
        
        # Make prediction
        if model is not None:
            prediction = model.predict(input_array)[0]
            prediction_label = "Mine" if prediction == 1 else "Rock"
            confidence = None
            
            # Try to get prediction probability if the model supports it
            try:
                confidence = model.predict_proba(input_array)[0]
                confidence = float(confidence[1]) if prediction == 1 else float(confidence[0])
                confidence = round(confidence * 100, 2)
            except:
                pass
            
            result = {
                "prediction": prediction_label,
                "confidence": confidence
            }
            return jsonify(result)
        else:
            return jsonify({"error": "Model not loaded"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_data', methods=['GET'])
def generate_data():
    try:
        # Generate 60 random values between 0 and 1
        random_data = np.random.uniform(0, 1, 60).tolist()
        return jsonify({"data": random_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run the app
    app.run(debug=True)