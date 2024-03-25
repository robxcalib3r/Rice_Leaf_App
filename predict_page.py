import streamlit as st
import numpy as np
from predict import predict
from PIL import Image
import tensorflow as tf
import zipfile
import tempfile
import os

class predict_page():
    def __init__(self) -> None:
        self.model = []
        self.class_names = ['BACTERIAL BLIGHT', 'BLAST', 'BROWNSPOT', 'TUNGRO']
        self.picture = None


    def upload_model(stream):
        if stream is not None:
            myzipfile = zipfile.ZipFile(stream)
            with tempfile.TemporaryDirectory() as tmp_dir:
                myzipfile.extractall(tmp_dir)
                root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
                model_dir = os.path.join(tmp_dir, root_folder)
                #st.info(f'trying to load model from tmp dir {model_dir}...')
                model = tf.keras.models.load_model(model_dir)


    def show_predict_page(self):
        st.title("Rice Leaf App")
        st.write("""### Give some rice leaf images to process """)

        # st.write("""#### Give your preferred model (or Default)""")
        # st.write("""(Currently only .keras format is supported)""")
        
        # bUpModel = st.button("Upload Model (.keras)")
        # bDefaultModel = st.button("Default Rice Leaf Model")

        # if bDefaultModel:
        try:
            self.model = tf.keras.models.load_model('rice_leaf_model.keras')
        except Exception as e:
            st.write(f""" ##### Model Not found! \n debug: {e}""")
        

        # elif bUpModel:
        #     try:
        #         file = st.file_uploader("Upload keras file", 'keras')
        #         self.model = self.upload_model(file)
        #     except Exception as e:
        #         st.write(f"Some error occurred! debug: {e}")
        
        
        st.write("""#### Upload picture or take a picture""")
        
        
        # picture = st.camera_input("Take a picture of infected Rice leaf")
        # if picture is not None:
        #     img = Image.open(self.picture)   # Read image buffer as PIL image
        #     img_np = np.array(img)      # To convert PIl image to Numpy Array

        #     predicted_class, confidence = predict(self.model, img_np, self.class_names)
        #     st.image(img_np)
        #     st.write(f"Predicted Class: {predicted_class}.\n Confidence: {confidence}%")

        
        picture_up = st.file_uploader("Choose Image and upload")
        if picture_up is not None:
            print('entered')
            img = Image.open(picture_up)   # Read image buffer as PIL image
            img_np = np.array(img)      # To convert PIl image to Numpy Array

            predicted_class, confidence = predict(self.model, img_np, self.class_names)
            st.image(img_np)
            st.write(f"Predicted Class: {predicted_class} ")
            st.write(f"Confidence: {confidence}%")
    
        


    
    