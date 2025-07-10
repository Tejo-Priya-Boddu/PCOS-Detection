import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
@st.cache_resource
def load_model():
    # Replace 'path_to_your_model.h5' with your actual model path
    model = tf.keras.models.load_model('pcos_detection_model.h5')
    return model

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app
def main():
    st.title("PCOS Detection from Ultrasound Images")
    st.write("Upload an ultrasound image to detect PCOS.")

    # Load model
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Analyzing the image...")
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Display results
        if predicted_class == 1:
            st.success("The image is classified as Normal (No PCOS detected).")
        else:
            st.warning("The image indicates PCOS detected. Please consult a healthcare provider.")


if __name__ == "_main_":
    main()