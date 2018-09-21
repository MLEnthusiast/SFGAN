# SFGAN
Semantic Fusion GAN for semi-supervised image classification

Code for paper **Semantic-Fusion GANs for Semi-Supervised Satellite Image Classification** accepted in the International Conference on Image Processing (**ICIP**) to be held in *Athens, Greece* in October, 2018.

Code is **available now**.

## Instructions
1. First download the [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip) data set and extract the images.
2. Run the file_reader.m to convert the images into a .mat file. This will be used as input for training the network.
3. Run sfgan_train_eval.py to train the network.

N.B. Python 3 is recommended for running this code as the batching gives errornoues results with lower versions of Python.

A commented version of the code will be updated soon.

