import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model

# Define directories for flower images
tulip_dir = os.path.join(r'D:\flowers\tulip')
sf_dir = os.path.join(r'D:\flowers\sunflower')
rose_dir = os.path.join(r'D:\flowers\rose')
dandelion_dir = os.path.join(r'D:\flowers\dandelion')
daisy_dir = os.path.join(r'D:\flowers\daisy')

# List and print sample images from each category
train_tulip_names = os.listdir(tulip_dir)
print(train_tulip_names[:5])

train_sf_names = os.listdir(sf_dir)
print(train_sf_names[:5])

# Set batch size
batch_size = 16

# Create ImageDataGenerator for training data
train_datagen = ImageDataGenerator(rescale=1./255)

# Create training data generator
train_generator = train_datagen.flow_from_directory(
    r'D:\flowers',
    target_size=(300, 300),
    batch_size=batch_size,
    color_mode='grayscale',
    classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    class_mode='categorical'
)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Print model summary
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['acc']
)

# Calculate total samples
total_sample = train_generator.n

# Set number of epochs
num_epochs = 5

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=int(total_sample / batch_size),
    epochs=num_epochs,
    verbose=1
)

# Save model architecture to JSON
model_json = model.to_json()
with open("modelGG.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights('modelGG.h5')
print("Saved model to disk")





from tensorflow.keras.models import model_from_json

# Load the model architecture
with open('modelGG.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights('modelGG.h5')
print("Loaded model from disk")



import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(300, 300), color_mode='grayscale')
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Rescale the image
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape (1, 300, 300, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    # Make predictions
    prediction = model.predict(img_array)
    # Get the class label
    class_idx = np.argmax(prediction, axis=1)[0]
    class_labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    return class_labels[class_idx]

# Example usage
img_path = 'path_to_your_image.jpg'  # Replace with the path to your image
predicted_class = predict_image(img_path)
print(f'The predicted class is: {predicted_class}')
