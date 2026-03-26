import streamlit as st
import os
import pickle
from PIL import Image
import tensorflow

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from annoy import AnnoyIndex

import numpy as np
# Load embeddings and filenames
feature_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Convert feature list to numpy array
feature_list = np.array(feature_list)

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# -----------------------------
# Build ANNOY Index
# -----------------------------
f = feature_list.shape[1]   # dimension of feature vectors

annoy_index = AnnoyIndex(f, 'euclidean')

for i in range(len(feature_list)):
    annoy_index.add_item(i, feature_list[i])

annoy_index.build(10)  # number of trees

# -----------------------------
# Save uploaded file
# -----------------------------
def save_upload_file(upload_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        with open(os.path.join('uploads', upload_file.name), 'wb') as f:
            f.write(upload_file.getbuffer())
        return True
    except:
        return False

# -----------------------------
# Feature Extraction
# -----------------------------
def feature_extraction(img_path, model):

    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# -----------------------------
# Recommendation using ANNOY
# -----------------------------
def recommend(features):

    indices = annoy_index.get_nns_by_vector(features, 5)

    return indices


# -----------------------------
# Streamlit File Upload
# -----------------------------
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:

    if save_upload_file(uploaded_file):

        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image,width=300)

        # Extract features
        features = feature_extraction(
            os.path.join("uploads", uploaded_file.name),
            model
        )

        # Get recommendations
        indices = recommend(features)
        st.subheader("Recommended Products")     # Display recommended images
        col1, col2, col3, col4, col5 = st.columns(5)


        with col1:
            st.image(filenames[indices[0]], width=300)

        with col2:
            st.image(filenames[indices[1]], width=300)

        with col3:
            st.image(filenames[indices[2]], width=300)

        with col4:
            st.image(filenames[indices[3]], width=300)

        with col5:
            st.image(filenames[indices[4]], width=300)


    else:
        st.header("Some error occurred in file upload")