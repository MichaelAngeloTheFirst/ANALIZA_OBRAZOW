import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import matplotlib.image as img
from tensorflow import keras



batch_size = 32
height, width = (256,256)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'train',
    seed=100,
    labels="inferred",
    label_mode="categorical",
    image_size = (height, width), 
    batch_size = batch_size,
    subset = 'training',
    validation_split=0.2,
    shuffle = True,
    color_mode="rgb",
    class_names=("Deinopis_Spider","Red_Knee_Tarantula", "Peacock_Spider")
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'train',
    seed=100,
    labels="inferred",
    label_mode="categorical",
    image_size = (height, width), 
    batch_size = batch_size,
    subset = 'validation',
    validation_split=0.2,
    shuffle = True,
    color_mode="rgb",
    class_names=("Deinopis_Spider","Red_Knee_Tarantula", "Peacock_Spider")
)


model = keras.models.Sequential()

initializers = {

}
model.add(
    keras.layers.RandomFlip("horizontal_and_vertical") )
model.add(
    keras.layers.RandomRotation(0.2)
)
model.add(
    keras.layers.Rescaling(1/127.0, offset=-1)
)
model.add( 
    keras.layers.Conv2D(
        24, 5, input_shape=(256,256,3), 
        activation='relu', 
    )
)
model.add( keras.layers.MaxPooling2D(2) )
model.add( 
    keras.layers.Conv2D(
        48, 5, activation='relu', 
    )
)
model.add( keras.layers.MaxPooling2D(2) )
model.add( 
    keras.layers.Conv2D(
        96, 5, activation='relu', 
    )
)
model.add( keras.layers.Flatten() )
model.add( keras.layers.Dropout(0.9) )

model.add( keras.layers.Dense(
    3, activation='softmax',
    )
)

# model.summary()


model.compile(optimizer = "adam", 
                loss='categorical_crossentropy',
                metrics=['acc'])


history = model.fit(
    train_dataset, 
    validation_data = val_dataset,
    workers=10,
    epochs=20,
    batch_size = batch_size
)



####

model.save("our_model.model")

# import cv2
# # type(plt.imread("Red_Knee_Tarantula/red.069.jpg"))
# temp = np.empty((1,256,256,3))
# temp[0]=cv2.resize(plt.imread("train/Brach.jpg"),(256,256),interpolation= cv2.INTER_NEAREST)

# # plt.imshow(temp[0][:,:,::-1])
# print(model.predict([temp]))

# val_los , val_acc = model.evaluate(val_dataset)
# print(val_los, val_acc)

# loss: 0.1834 - acc: 0.9299 