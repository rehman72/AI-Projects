import cv2 
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
from PIL import Image

def load_model():
    model=MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img=np.array(image)
    img=cv2.resize(img,(224,224))
    img=preprocess_input(img)
    img=np.expand_dims(img,axis=0)
    return img

def classify_image(model,image):
    try:
        processed_image=preprocess_image(image)
        predictions=model.predict(processed_image)
        decoded_predictions=decode_predictions(predictions,top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"An error occurred during image classification: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Image Classifier",page_icon="🖼️",layout="centered")
    st.title("AI Image Classifier 📸")
    st.write("Upload an image, and the AI will classify it for you!")


    def load_cached_model():
        return load_model()
    
    model=load_cached_model()
    upload_file=st.file_uploader("Upload an Image",type=["jpg","jpeg","png"])

    if upload_file is not None:
        image=st.image(upload_file,caption="Uploaded Image",use_container_width=True)
        btn=st.button("Classify Image")

        if btn:
            with st.spinner("Classifying..."):
                image=Image.open(upload_file)
                predictions=classify_image(model,image)
                if predictions:
                    st.subheader("Predictions")
                    for _, label,score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__=="__main__":
    main()

