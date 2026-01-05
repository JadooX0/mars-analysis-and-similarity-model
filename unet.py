import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

size = 128
mask = np.zeros((size, size), dtype=np.float32)
mask[20:100, 20:100] = 1.0 

image_path = "/home/navya/Downloads/data/img5.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (size, size)) / 255.0

X = np.expand_dims(np.expand_dims(img, 0), -1)
y = np.expand_dims(np.expand_dims(mask, 0), -1)


inputs = Input((size, size, 1))
c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
p1 = MaxPooling2D((2, 2))(c1)
bn = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(bn)
m1 = concatenate([u1, c1])
outputs = Conv2D(1, (1, 1), activation='sigmoid')(m1)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy')


model.fit(X, y, epochs=150, verbose=0)


pred = model.predict(X)
raw_pred = pred[0].squeeze()

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Input Mars Image")
plt.imshow(img, cmap='gray')

plt.subplot(1, 4, 2)
plt.title("Target Mask")
plt.imshow(mask, cmap='gray')

plt.subplot(1, 4, 3)
plt.title("Confidence Heatmap")

plt.imshow(raw_pred, cmap='jet') 
plt.colorbar()

plt.subplot(1, 4, 4)
plt.title("Thresholded (0.4)")
plt.imshow(raw_pred > 0.4, cmap='gray')

plt.show()