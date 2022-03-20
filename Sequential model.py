import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

sample_image = image.load_img("C:/Users/Admin/PycharmProjects/Prediction model using gabble/basedata/training_set/cats/cat.1.jpg")
plt.imshow(sample_image)

print(cv2.imread("C:/Users/Admin/PycharmProjects/Prediction model using gabble/basedata/training_set/cats/cat.1.jpg").shape)
train = ImageDataGenerator(rescale= 1/40)
validation = ImageDataGenerator(rescale=1/40)
train_dataset = train.flow_from_directory('C:/Users/Admin/PycharmProjects/Prediction model using gabble/basedata/training_set/',
                                          target_size = (200,200),
                                          batch_size = 100,
                                          class_mode = 'binary')
validation_dataset = validation.flow_from_directory('C:/Users/Admin/PycharmProjects/Prediction model using gabble/basedata/validation_set/',
                                          target_size = (200,200),
                                          batch_size = 100,
                                          class_mode = 'binary')

print(train_dataset.class_indices)
print(train_dataset.classes)

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu',),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    ##
                                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                                    ])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.01),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 5,
                      epochs = 1601,
                      validation_data = validation_dataset)

dir_path = 'C:/Users/Admin/PycharmProjects/Prediction model using gabble/basedata/training_set'
for i in os.listdir(dir_path):
    img = image.load_img(dir_path + "/" + i, target_size=(200,200,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([X])

    val = model.predict(images)
    if val == 0:
        print("Cat!")
    else:
        print("Dog!")
