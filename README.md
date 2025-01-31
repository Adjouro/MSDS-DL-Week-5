# MSDS-DL-Week-5 (The Notebook was to big to be upload on Github, so I uploaded screenshots, and the .py version)

Use your training skills to create images, rather than identify them. You’ll be using GANs, which are at the creative frontier of machine learning. You might think of GANs as robot artists in a sense—able to create eerily lifelike images, and even digital worlds In this competition, you’ll be training generative models to create images of dogs. Only this time… there’s no ground truth data for you to predict. Here, you’ll submit the images and be scored based on how well those images are classified as dogs from pre-trained neural networks

## EDA Results
The Dataset has been well load. No duplications. It contains 20K images distributed accros 120 breed categories. Images vary widely in width and height, ranging from small (under 500px) to large (3000px+) but the dataset is well balanced accross breed. Resizing will be necessary for training.

## Conclusion
The goal of this project was to train a DCGAN to generate realistic images of dogs using the "Generative Dog Images" dataset.

The dataset was loaded, preprocessed (resizing, augmentation). A DCGAN model was implemented with convolutional layers. The model was trained for 100 epochs. Loss curves and generated images were analyzed.


The DCGAN successfully generated images that contain abstract "representations" of dogs.However, the images remain blurry and lack clear structure. The loss curves indicate that the model is still improving, but additional training is needed but due to a lack of ressources and times we can't proceed the training phase.



if we had more ressources available we could:

Increase training time to 300+ epochs.
Use larger image sizes (128x128 instead of 64x64).
Tune hyperparameters (lower learning rate).




