import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt
import pathlib
tfds.disable_progress_bar()

print(tf.__version__)

splits, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True,
                         split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

class_names = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

with open('labels.txt', 'w') as f:
    f.write('\n'.join(class_names))

IMG_SIZE = 28


# Write a function to normalize and resize the images

def format_example(image, label):
    # Cast image to float32
    image = tf.cast(image, tf.float32)
    # Resize the image if necessary
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize the image in the range [0, 1]
    image = image / 255.0
    return image, label


# Set the batch size to 32
BATCH_SIZE = 32

# Prepare the examples by preprocessing them and then batching them (and optionally prefetching them)

# If you wish you can shuffle train set here
train_batches = train_examples.cache().shuffle(num_examples // 4).batch(BATCH_SIZE).map(format_example).prefetch(1)
validation_batches = validation_examples.cache().batch(BATCH_SIZE).map(format_example).prefetch(1)
test_batches = test_examples.cache().batch(1).map(format_example)

# Build the model shown in the previous cell


model = tf.keras.Sequential([
    # Set the input shape to (28, 28, 1), kernel size=3, filters=16 and use ReLU activation,
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Set the number of filters to 32, kernel size to 3 and use ReLU activation
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # Flatten the output layer to 1 dimension
    tf.keras.layers.Flatten(),
    # Add a fully connected layer with 64 hidden units and ReLU activation
    tf.keras.layers.Dense(64, activation='relu'),
    # Attach a final softmax classification head
    tf.keras.layers.Dense(10, activation='softmax')])

# Set the loss and accuracy metrics
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

validation_batches
train_batches,

model.fit(train_batches, epochs=15, validation_data=validation_batches)

export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)

optimization = tf.lite.Optimize.DEFAULT

# Use the TFLiteConverter SavedModel API to initialize the converter
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

# Set the optimzations
converter.optimizations = [optimization]

# Invoke the converter to finally generate the TFLite model
tflite_model = converter.convert()

tflite_model_file = 'model.tflite'

with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Gather results for the randomly sampled test images
predictions = []
test_labels = []
test_images = []

for img, label in test_batches.take(50):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_labels.append(label[0])
    test_images.append(np.array(img))


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label.numpy():
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(list(range(10)), class_names, rotation='vertical')
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array[0])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


# @title Visualize the outputs { run: "auto" }
index = 12  # @param {type:"slider", min:1, max:50, step:1}
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(index, predictions, test_labels, test_images)
plt.show()
plot_value_array(index, predictions, test_labels)
plt.show()
