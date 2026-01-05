import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity


size = 128
inputs = Input((size, size, 1))
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2, 2))(c1)
bn = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(bn)
m1 = concatenate([u1, c1])
outputs = Conv2D(1, (1, 1), activation='sigmoid')(m1)

model = Model(inputs, outputs)

feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=3).output)

def get_features(img_path):
    if not os.path.exists(img_path):
        return np.zeros((1, 32*32*32)) 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size)) / 255.0
    img = np.expand_dims(np.expand_dims(img, 0), -1)
    feat = feature_extractor.predict(img, verbose=0)
    return feat.flatten().reshape(1, -1)


mars_path = "/home/navya/Downloads/data/img5.png"
earth_path = "/home/navya/Downloads/data/img6.png" 

mars_vec = get_features(mars_path)
earth_vec = get_features(earth_path)

similarity = cosine_similarity(mars_vec, earth_vec)[0][0]

print(f"\n" + "="*30)
print(f"SIMILARITY SCORE: {similarity:.4f}")
print("="*30)