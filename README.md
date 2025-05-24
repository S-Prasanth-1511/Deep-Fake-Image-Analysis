# Deep Fake Image Analysis

A machine learning web application to detect AI-generated (deepfake) images using CNN-based eye-region gradient analysis. Built with Python, Flask, TensorFlow, and OpenCV, the system classifies uploaded images as real or fake and displays prediction confidence in real-time.

## 🔍 Features
- Upload interface for real-time fake image detection
- Gradient-based analysis focused on eye region
- CNN-based model for high accuracy classification
- Displays confidence score for each prediction
- Simple and interactive Flask web UI

## 🛠️ Tech Stack
- **Backend**: Python, Flask, TensorFlow, Keras
- **Image Processing**: OpenCV
- **Frontend**: HTML, CSS (via Flask templates)


### ✅ Prerequisites
- Python 3.x
- pip (Python package installer)

### 📦 Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/deepfake-image-analysis.git
   cd deepfake-image-analysis
   
**2.Install dependencies**
bash
Copy
Edit
pip install -r requirements.txt

**3.Run the application**
bash
Copy
Edit
python app.py

**4.Access the app**
Open your browser and go to:
http://localhost:5000
