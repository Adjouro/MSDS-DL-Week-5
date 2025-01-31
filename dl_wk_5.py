# -*- coding: utf-8 -*-
"""dl-wk-5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Smf-8fvwvjnTlATs1DQugApQNsbLW_vG

# Generative Dog Images

Use your training skills to create images, rather than identify them. You’ll be using GANs, which are at the creative frontier of machine learning. You might think of GANs as robot artists in a sense—able to create eerily lifelike images, and even digital worlds
In this competition, you’ll be training generative models to create images of dogs. Only this time… there’s no ground truth data for you to predict. Here, you’ll submit the images and be scored based on how well those images are classified as dogs from pre-trained neural networks

## Load Data
"""

from google.colab import drive
drive.mount('/content/drive')

#Initial libraries
import zipfile
import os
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np

# Define paths
dogs_zip_path = "/content/drive/MyDrive/Colab Notebooks/dog_data/all-dogs.zip"
annotations_zip_path = "/content/drive/MyDrive/Colab Notebooks/dog_data/Annotation.zip"

# Define extraction folders
extract_path = "/content/dog_data_unziped"
os.makedirs(extract_path, exist_ok=True)  # Create directory if not exists

# Function to extract zip files
def unzip_file(zip_path, destination):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

# Extract files
unzip_file(dogs_zip_path, extract_path)
unzip_file(annotations_zip_path, extract_path)

print("Extraction complete!")

# Define paths after extraction
dogs_path = "/content/dog_data_unziped/all-dogs"
annotations_path = "/content/dog_data_unziped/Annotation"

# List some files
print("All Dogs Sample:", os.listdir(dogs_path)[:5])  # Show first 5 images
print("Annotations Sample:", os.listdir(annotations_path)[:5])  # Show first 5 breed folders

"""# Exploratory Data Analysis"""

# Select a random image
random_image = random.choice(os.listdir(dogs_path))
img_path = os.path.join(dogs_path, random_image)

# Read and display the image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.title(random_image)
plt.show()

# Check number of images
num_images = len(os.listdir(dogs_path))
print(f"Total images in 'all-dogs': {num_images}")

# Check number of breed folders in Annotations
num_breeds = len(os.listdir(annotations_path))
print(f"Total breed categories in 'Annotation': {num_breeds}")

image_sizes = []

for img_file in os.listdir(dogs_path):
    img_path = os.path.join(dogs_path, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        height, width, _ = img.shape
        image_sizes.append((width, height))

# Convert to numpy array
image_sizes = np.array(image_sizes)

# Plot histogram of image sizes
plt.figure(figsize=(10, 5))
plt.scatter(image_sizes[:, 0], image_sizes[:, 1], alpha=0.5)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution")
plt.show()

from PIL import Image

# Check for corrupt images
corrupt_images = []

for img_file in os.listdir(dogs_path):
    img_path = os.path.join(dogs_path, img_file)
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify image integrity
    except Exception as e:
        corrupt_images.append(img_file)

print(f"Number of corrupt images: {len(corrupt_images)}")
if corrupt_images:
    print("Corrupt images:", corrupt_images)

import collections

# Count images per breed in Annotations
breed_counts = {breed: len(os.listdir(os.path.join(annotations_path, breed))) for breed in os.listdir(annotations_path)}

# Sort and plot
sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(15,5))
plt.bar([b[0] for b in sorted_breeds[:20]], [b[1] for b in sorted_breeds[:20]])
plt.xticks(rotation=90)
plt.ylabel("Number of Images")
plt.title("Top 20 Dog Breeds with Most Images")
plt.show()

"""### Exploratory Data Analysis Conclusion

The Dataset has been well load. No duplications. It contains 20K images distributed accros 120 breed categories. Images vary widely in width and height, ranging from small (under 500px) to large (3000px+) but the dataset is well balanced accross breed. Resizing will be necessary for training.

## Pre-processing

### Data resizing
"""

import cv2
import os

# Target size
img_size = 64  # Change to 128 if needed
save_dir = "/content/dog_data_resized"

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

for img_file in os.listdir(dogs_path):
    img_path = os.path.join(dogs_path, img_file)
    save_path = os.path.join(save_dir, img_file)

    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))  # Resize
        cv2.imwrite(save_path, img)  # Save resized image

print("All images resized successfully!")

"""### Data Augmentation"""

import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

"""### Normalise the Dataset"""

import tensorflow as tf
import glob

# Define dataset path
dataset_path = "/content/dog_data_resized"
image_size = 64  # Use 64x64 images
batch_size = 64

# Load images and normalize pixel values to [-1, 1]
def load_dataset():
    data = []
    for img_path in glob.glob(os.path.join(dataset_path, "*.jpg")):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_size, image_size))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = (img - 127.5) / 127.5  # Normalize to range [-1, 1]
        data.append(img)
    return np.array(data)

dataset = load_dataset()
print(f"Dataset shape: {dataset.shape}")  # Expected: (num_images, 64, 64, 3)

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(10000).batch(batch_size)

"""## Build the Model : DCGAN"""

from tensorflow.keras import layers

def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),  # Reshape to (8,8,256)

        layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation='tanh')
    ])

    return model

generator = build_generator()
generator.summary()

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

discriminator = build_discriminator()
discriminator.summary()

# Loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

"""## Train the Model"""

import time
import numpy as np

epochs = 100
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])  # Fixed seed for visualization

# Track losses
gen_losses = []
disc_losses = []

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss  # Return losses for tracking


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        total_gen_loss = 0
        total_disc_loss = 0
        batch_count = 0

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            total_gen_loss += gen_loss.numpy()
            total_disc_loss += disc_loss.numpy()
            batch_count += 1

        # Compute average losses for the epoch
        avg_gen_loss = total_gen_loss / batch_count
        avg_disc_loss = total_disc_loss / batch_count

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        # Generate images for visualization
        generate_and_save_images(generator, epoch + 1, seed)

        print(f"Epoch {epoch+1}: Gen Loss = {avg_gen_loss:.4f}, Disc Loss = {avg_disc_loss:.4f} (Time: {time.time() - start:.2f}s)")

    generate_and_save_images(generator, epochs, seed)

    # After training, plot loss curves
    plot_loss_curve(gen_losses, disc_losses)


def plot_loss_curve(gen_losses, disc_losses):
    """Plots generator & discriminator loss curves over epochs."""
    plt.figure(figsize=(10,5))
    plt.plot(gen_losses, label="Generator Loss")
    plt.plot(disc_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator & Discriminator Loss Over Time")
    plt.legend()
    plt.show()


# Start training with loss tracking
train(train_dataset, epochs)

"""## Try to generate Dog images..."""

import matplotlib.pyplot as plt
import tensorflow as tf

def generate_final_images(num_images=16):
    noise_dim = 100  # Keep consistent with training
    noise = tf.random.normal([num_images, noise_dim])  # Generate random noise

    generated_images = generator(noise, training=False)  # Generate images

    plt.figure(figsize=(8,8))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        plt.imshow((generated_images[i].numpy() + 1) / 2)  # Normalize from [-1,1] to [0,1]
        plt.axis('off')

    plt.show()

generate_final_images()

"""## Conclusion

The goal of this project was to train a DCGAN to generate realistic images of dogs using the "Generative Dog Images" dataset.

The dataset was loaded, preprocessed (resizing, augmentation).
A DCGAN model was implemented with convolutional layers. The model was trained for 100 epochs. Loss curves and generated images were analyzed.

\
The DCGAN successfully generated images that contain abstract "representations" of dogs.However, the images remain blurry and lack clear structure.
The loss curves indicate that the model is still improving, but additional training is needed but due to a lack of ressources and times we can't proceed the training phase.

\

if we had more ressources available we could:
- Increase training time to 300+ epochs.
- Use larger image sizes (128x128 instead of 64x64).
- Tune hyperparameters (lower learning rate).


"""

