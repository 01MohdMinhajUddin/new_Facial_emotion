# Facial Emotion Recognition – Streamlit and TensorFlow

This project is a facial emotion recognition web app built with TensorFlow / Keras and Streamlit.

Given an image of a person, the app detects the face, crops it, and predicts one of five emotions:

- Angry
- Happy
- Neutral
- Sad
- Surprise

The convolutional neural network (CNN) model was built entirely from scratch.
No transfer learning and no pre-trained backbone models (such as VGG16, ResNet, or MobileNet) were used.

The app uses a Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect faces and crop them before sending the image to the model. This makes the predictions focus on the face instead of the full background.

--------------------------------------------------
1. Project Overview
--------------------------------------------------

- Task: Image classification for facial emotions
- Model: Custom CNN built from scratch (no transfer learning)
- Frontend: Streamlit web application
- Main libraries: TensorFlow, Keras, OpenCV, NumPy, Pillow, Streamlit

Input: JPG, JPEG, or PNG image containing a face  
Output: Predicted emotion label and confidence score

--------------------------------------------------
2. Dataset
--------------------------------------------------

The dataset is stored in an "images" folder with one subfolder per class:

images/
  Angry/
  Happy/
  Neutral/
  Sad/
  Surprise/

Each subfolder contains face images for that emotion.

The dataset is imbalanced (fewer images for Angry and Surprise, more for Neutral, Sad, and Happy).
To handle this imbalance during training, the model uses:

- Data augmentation
- Class weights computed from the class distribution

The train and validation split is handled using Keras ImageDataGenerator with validation_split = 0.2, so each class is split into 80% training and 20% validation.

--------------------------------------------------
3. Model and Training
--------------------------------------------------

Model type:
- Custom CNN built completely from scratch (no transfer learning)
- Input size: 128 x 128 x 3

High-level architecture:

1. Several convolution blocks:
   - Conv2D layers with ReLU activation
   - BatchNormalization
   - MaxPooling2D
   - Dropout for regularization

2. Flatten layer

3. Fully connected Dense layers

4. Output Dense layer with 5 units and softmax activation
   - One unit for each emotion: Angry, Happy, Neutral, Sad, Surprise

Preprocessing during training:

- Images resized to 128 x 128
- Pixel values scaled to the range [0, 1] using rescale = 1.0 / 255
- Data augmentation on the training set:
  - Rotation
  - Zoom
  - Horizontal flip
  - Small shifts
- Class weights calculated from class counts to give more importance to minority classes

Callbacks:

- EarlyStopping to stop training when validation loss stops improving and restore the best weights
- ReduceLROnPlateau to reduce the learning rate when validation loss plateaus

Training results (summary):

- Best validation accuracy around 75–78%
- Validation loss around 0.6 at the best epoch
- Early stopping restores the model weights from the best epoch

--------------------------------------------------
4. Face Detection and Cropping (Haar Cascade)
--------------------------------------------------

For uploaded images, the app runs the following pipeline:

1. Load the image using PIL and convert to RGB:
   - Image.open(image_path).convert("RGB")
2. Convert the image to a NumPy array.
3. Convert the image to grayscale using OpenCV:
   - gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
4. Load the Haar Cascade classifier:
   - face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
   - faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
5. If one or more faces are detected:
   - Choose the largest face based on area (width * height).
   - Crop that region from the original RGB image:
     - face_rgb = img_np[y:y+h, x:x+w]
6. If no faces are detected:
   - Use the full image as a fallback.
7. Resize the face crop to 128 x 128.
8. Scale pixel values to the range [0, 1].
9. Add a batch dimension so the final shape is (1, 128, 128, 3) and send it to the model.

This Haar Cascade based cropping is used in the app so the CNN always sees a focused face region rather than the entire background.

--------------------------------------------------
5. Project Structure
--------------------------------------------------

Example project layout:

```text
new_facial_emotion/
  .venv/                              # local virtual environment (not committed)
  images/                             # dataset (one folder per emotion)
  app.py                              # Streamlit app
  new_facial_emotion.h5               # trained Keras model
  haarcascade_frontalface_default.xml # Haar Cascade for face detection
  working.ipynb                       # training and experiments notebook
  requirements.txt                    # project dependencies
  README.md                           # project description

```


--------------------------------------------------
6. How to Run Locally
--------------------------------------------------

6.1 Clone the repository

git clone https://github.com/01MohdMinhajUddin/new_Facial_emotion.git
cd new_facial_emotion

6.2 Create and activate a virtual environment (Windows)

python -m venv .venv
.venv\Scripts\activate

On macOS or Linux:

python -m venv .venv
source .venv/bin/activate

6.3 Install dependencies

pip install -r requirements.txt

6.4 Run the Streamlit app

streamlit run app.py

Then open the URL shown in the terminal (usually http://localhost:8501).

--------------------------------------------------
7. Using the Web App
--------------------------------------------------

1. Upload a JPG, JPEG, or PNG image using the file uploader.
2. The app displays the uploaded image.
3. The Haar Cascade detects and crops the face.
4. The CNN model predicts the emotion of the face.
5. The app shows:
   - Emotion: the predicted class label
   - Confidence: the model probability in percent

If the predicted confidence is below a chosen threshold (for example 35%), the app displays a message that the model is not very confident in its prediction.

--------------------------------------------------
8. Deployment Notes
--------------------------------------------------

The app is designed to be deployed as a simple Streamlit service.

A typical deployment flow:

1. Push this project to GitHub.
2. Make sure the repository contains:
   - app.py
   - new_facial_emotion.h5
   - haarcascade_frontalface_default.xml
   - requirements.txt
   - Procfile (for platforms like Heroku)

3. On Heroku or another platform, configure the Procfile to run:

web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

4. Deploy and open the app URL provided by the platform.

--------------------------------------------------
9. Possible Future Improvements
--------------------------------------------------

- Train on a larger and more diverse facial emotion dataset.
- Add more detailed evaluation metrics, such as per-class precision, recall, F1 score, and a confusion matrix.
- Improve handling of images with multiple faces by showing predictions for each detected face.
- Add live webcam support for real-time emotion detection.
- In a separate project, experiment with transfer learning models and compare them against this from-scratch CNN.
