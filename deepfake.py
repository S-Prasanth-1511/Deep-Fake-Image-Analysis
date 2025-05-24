import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded'

# Load your trained model
model = load_model("eye_gradient_AI_classifier.h5")

# Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
img_size = 64

def compute_combined_gradient(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) < 2:
        return None, None

    gradients = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_roi = image[ey:ey+eh, ex:ex+ew]
        b, g, r = cv2.split(eye_roi)

        def grad(c):
            gx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(gx, gy)

        grad_r, grad_g, grad_b = grad(r), grad(g), grad(b)
        combined_grad = (grad_r + grad_g + grad_b) / 3.0
        combined_grad_resized = cv2.resize(combined_grad, (img_size, img_size))
        gradients.append(combined_grad_resized)

    if len(gradients) == 2:
        return gray, np.stack(gradients, axis=-1)
    else:
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_images = []
    prediction = None
    prediction_score = None
    original_image = None
    gray_image = None

    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return render_template('index.html', error="No file uploaded")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Convert and save grayscale image
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_' + filename)
        cv2.imwrite(gray_path, gray)

        original_image = 'uploaded/' + filename
        gray_image = 'uploaded/' + 'gray_' + filename

        # Compute gradient and predict
        gray, processed_features = compute_combined_gradient(filepath)
        if processed_features is not None:
            processed_features_input = np.expand_dims(processed_features, axis=0) / 255.0
            raw_prediction = model.predict(processed_features_input)
            prediction_score = float(raw_prediction[0][0])
            prediction = "Fake" if prediction_score < 0.5 else "Real"

            # Save gradient images
            left_grad = (processed_features[:, :, 0] * 255).astype(np.uint8)
            right_grad = (processed_features[:, :, 1] * 255).astype(np.uint8)

            # Resize to larger display size (e.g., 256x256)
            left_grad_large = cv2.resize(left_grad, (256, 256), interpolation=cv2.INTER_CUBIC)
            right_grad_large = cv2.resize(right_grad, (256, 256), interpolation=cv2.INTER_CUBIC)

            left_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradient_left.jpg')
            right_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradient_right.jpg')
            cv2.imwrite(left_path, left_grad_large)
            cv2.imwrite(right_path, right_grad_large)


            uploaded_images.extend([
                'uploaded/gradient_left.jpg',
                'uploaded/gradient_right.jpg'
            ])

    return render_template('index.html',
                           uploaded_images=uploaded_images,
                           prediction=prediction,
                           prediction_score=prediction_score,
                           original_image=original_image,
                           gray_image=gray_image)

if __name__ == "__main__":
    app.run(debug=True)
