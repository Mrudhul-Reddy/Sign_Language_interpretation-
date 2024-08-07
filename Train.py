import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Define constants
new_image_size = (224,224)
batch_size = 32
epochs = 50
num_classes = 36  # Number of directories

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Load images from directories with the new target size
train_generator = train_datagen.flow_from_directory(
    'D:\Pycharm\ASL2H\Images',
    target_size=new_image_size,  # Change target_size to the new image size
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(new_image_size[0], new_image_size[1], 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the model
model.save('model.h5')

# Create a text file with labels
labels_file = open('labels.txt', 'w')
for label in range(num_classes):
    labels_file.write(f'{label}  {chr(ord("0") + label % 10) if label < 10 else chr(ord("A") + label % 26)}\n')
labels_file.close()