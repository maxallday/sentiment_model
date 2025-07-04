# This code demonstrates how to build, train, and evaluate a simple neural network using TensorFlow and Keras on the MNIST dataset.
# The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9).
# It is a classic dataset used for training image classification models.
# Import TensorFlow. TensorFlow provides the deep learning framework and Keras API for building models.
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset directly from TensorFlow's built-in datasets.
# The dataset is automatically downloaded from an online source and split into training and test sets.
# x_train and x_test contain image pixel data, while y_train and y_test contain the corresponding digit labels.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values of the images from the original range of 0 to 255 to a range of 0 to 1.
# This scaling helps the neural network train faster and more reliably.
x_train, x_test = x_train / 255.0, x_test / 255.0


# Display the first image in the training set
#plt.imshow(x_train[0], cmap='gray')
#plt.title(f"Label: {y_train[0]}")
#plt.axis('off')
#plt.show()

# Build a Sequential model, which is a linear stack of layers.
model = tf.keras.models.Sequential([
    # The Flatten layer converts the 2D 28x28 images into a 1D array of 784 pixels.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # A Dense (fully-connected) layer with 128 neurons and ReLU activation for introducing non-linearity.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer randomly sets 20% of its inputs to zero during training.
    # This prevents overfitting by ensuring that the model does not rely too heavily on any particular set of features.
    tf.keras.layers.Dropout(0.2),
    
    # The final Dense layer with 10 neurons and softmax activation.
    # Each neuron corresponds to one of the 10 digits (0-9), and softmax outputs a probability distribution.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model.
# - optimizer='adam': Adam optimizer adjusts the learning rate during training.
# - loss='sparse_categorical_crossentropy': This loss function is used for integer-labeled classification.
# - metrics=['accuracy']: The model will report accuracy during training and testing.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train (fit) the model on the training data over 5 epochs.
# An epoch means one full pass through the entire training dataset.
#model.fit(x_train, y_train, epochs=5),

# Evaluate the model on the test set.
# This provides an unbiased evaluation of how well the model generalizes to new, unseen data.
#model.evaluate(x_test, y_test)
#history stores the training history, including loss and accuracy for each epoch.
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# Evaluate the model on the test set to see how well it performs on unseen data.
model.evaluate(x_test, y_test)
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


