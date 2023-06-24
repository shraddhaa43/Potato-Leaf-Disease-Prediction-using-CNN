#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image 


# In[38]:


# generate batches of tensor image data with real-time data augmentation
datagen=ImageDataGenerator(rescale=1./255,rotation_range=30,horizontal_flip=True)
training_set=datagen.flow_from_directory(r"Project_Dataset\Training",
                                         target_size=(120,120),
                                         batch_size=300,
                                         class_mode='categorical')
test_set=datagen.flow_from_directory(r"Project_Dataset\Testing",
                                         target_size=(120,120),
                                         batch_size=300,
                                         class_mode='categorical')
class_labels={0:"Early_Blight",1:"Healthy",2:"Late_Blight"}


# In[3]:


model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(120,120,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128,(3,3)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()


# In[4]:


model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(training_set,epochs=60,validation_data=test_set)


# In[44]:


img = image.load_img(r"Project_Dataset\Validation\Early_Blight\Early_Blight_1.jpg",target_size=(120,120,3))
plt.imshow(img)

test_img=np.asarray(img)
test_img.shape

test_img=test_img.reshape(1,120,120,3)
yhat=model.predict(test_img)
predicted_class = np.argmax(yhat[0])
predicted_labels=class_labels[predicted_class]
print("Predicted Label is: ",predicted_labels)


# In[39]:


img = image.load_img(r"Project_Dataset\Validation\Healthy\Healthy_5.jpg",target_size=(120,120,3))
plt.imshow(img)

test_img=np.asarray(img)
test_img.shape

test_img=test_img.reshape(1,120,120,3)
yhat=model.predict(test_img)
predicted_class = np.argmax(yhat[0])
predicted_labels=class_labels[predicted_class]
print("Predicted Label is: ",predicted_labels)


# In[41]:


img = image.load_img(r"Project_Dataset\Validation\Late_Blight\Late_Blight_3.jpg",target_size=(120,120,3))
plt.imshow(img)

test_img=np.asarray(img)
test_img.shape

test_img=test_img.reshape(1,120,120,3)
yhat=model.predict(test_img)
predicted_class = np.argmax(yhat[0])
predicted_labels=class_labels[predicted_class]
print("Predicted Label is: ",predicted_labels)


# In[45]:


for i in range(1,164):
    img = image.load_img(f"Project_Dataset\Validation\Early_Blight\Early_Blight_{i}.jpg",target_size=(120,120,3))
    test_img=np.asarray(img)
    test_img=test_img.reshape(1,120,120,3)
    yhat=model.predict(test_img)
    predicted_class = np.argmax(yhat[0])
    predicted_labels=class_labels[predicted_class]
    print("Predicted Label is: ",predicted_labels)


# In[48]:


for i in range(1,60):
    img = image.load_img(f"Project_Dataset\Validation\Healthy\Healthy_{i}.jpg",target_size=(120,120,3))
    test_img=np.asarray(img)
    test_img=test_img.reshape(1,120,120,3)
    pre1.append(model.predict(test_img))
    yhat=model.predict(test_img)
    predicted_class = np.argmax(yhat[0])
    predicted_labels=class_labels[predicted_class]
    print("Predicted Label is: ",predicted_labels)


# In[49]:


for i in range(1,71):
    img = image.load_img(f"Project_Dataset\Validation\Late_Blight\Late_Blight_{i}.jpg",target_size=(120,120,3))
    test_img=np.asarray(img)
    test_img=test_img.reshape(1,120,120,3)
    yhat=model.predict(test_img)
    predicted_class = np.argmax(yhat[0])
    predicted_labels=class_labels[predicted_class]
    print("Predicted Label is: ",predicted_labels)

