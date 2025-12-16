import numpy as np
import tensorflow  as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import streamlit as st


classnames=['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


# ------------------------
#load model
# ------------------------

@st.cache_resource
def load_model():
    model=tf.keras.models.load_model("new_facial_emotion.h5")
    return model

model=load_model()


st.set_page_config(
    page_title="Facial_emotion_recognition",
    layout="centered"
)


def predict_image(image_path):

    # 1. Showing the full image
    original_image= Image.open(image_path).convert("RGB")
    original_image_resize= original_image.resize((460,560))

    # 2. Detect the face
    img_np = np.array(original_image)
    gray = cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray,1.1,5,minSize=(60,60))

    if len(faces)>0:
        x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
        face_rgb = img_np[y:y+h,x:x+w]
    else:
        face_rgb=img_np

    # 3. Resizing , normalizing and expanding dims
    face_resize = tf.image.resize(face_rgb,(128,128)) / 255.0
    image_array = tf.expand_dims(face_resize,axis=0)

    return original_image_resize ,image_array

# ------------------------
# UI
# ------------------------

st.title("Face Emotion Recognition")
st.write("This Project help you predict 5 Facial emotions ---> [ Surprise, Sad, Angry, Neutral, Happy ]")
uploaded_file=st.file_uploader(
    "upload an image [jpg, jpeg, png] ",
    type=["jpeg","jpg","png"]
)

if uploaded_file is not None:

    # Preprocess directly from uploaded_file
    rgb_image,input_tensor=predict_image(uploaded_file)

    # Show original image
    st.image(rgb_image,caption="Uploaded Image")

    # predict the image
    preds=model.predict(input_tensor)
    pred_idx=np.argmax(preds)
    pred_label=classnames[pred_idx]

    confidence=float(np.max(preds))

    if confidence < 0.30:
        st.write("Model is not confident")
        st.write(f"Emotion : {pred_label}")
    else:
        st.write(f"Emotion : {pred_label}")
    st.write(f"Confidence : {confidence*100:.2f} %")

else:
    st.info("Upload an image to Predict an Emotion")