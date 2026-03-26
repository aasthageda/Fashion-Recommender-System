import pickle
import tensorflow
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from app import feature_list, filenames
from sklearn.neighbors import NearestNeighbors
import cv2

git remote add originfeature_list = pickle.load(open('embeddings.pkl','rb'))

print(feature_list)
print(np.array(feature_list).shape)

filenames = pickle.load(open('filenames.pkl','rb'))


model =ResNet50(weights = 'imagenet' ,include_top = False ,input_shape =(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([model,
                                     GlobalMaxPooling2D()])

img = image.load_img('sample/i1.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors = 5 , algorithm = 'brute' ,metric = 'euclidean')
neighbors.fit(feature_list)

distance,indices = neighbors.kneighbors([normalized_result])
print(indices)

for file in indices[0]:
    print(filenames[file])
for file in indices[0]:
    temp_img = cv2.imread(filenames[file])

    if temp_img is None:
        print("Image not loaded:", filenames[file])
        continue

    resized_img = cv2.resize(temp_img, (512, 512))
    cv2.imshow('output', resized_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
