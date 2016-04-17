<h1> Digit Recognition using Neural Networks </h1>

This program presents a Neural Network capable of recognizing hand written digits.
The project was developed for one of my student projects namely Parallel computing. 
The goal is to create 2 version of a neural network, one that is trained serially (written in C++)
and one that is trained using one or more GPGPUs (written in CUDA), then prove the accerelation that HPC 
offers.

<h2> What to expect </h2>
This repo contains 2 branches, one for each version of the code. The serial version exists primarily for
benchmarking against the CUDA code.
I am aware of the many design and performance flaws of the current project and do not intend to 
spend more time on it for now. However i will maintain this repo primarily for personal reasons
(this is the first serious programming project i developed as a student, i would like to remember it).

<h2> Training set </h2>

The training set was downloaded from the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> database.
It consists of 60.000 sample images and their respective labels. Each image shows a digit ranging from 0-9.

<h2> Result </h2>
The network is capable of correctly predicting more than 90% even on unknown data 
(image samples that were not used during training phase) when training with more than 20000 images.
This should take at most 1 minute on any modern CPU. The CUDA version can be trained with the same dataset and 
therefore achieve the same results, in a fraction of that time, namely less than 10 seconds on a desktop GTX NVIDIA.

<h2> How to use it </h2>
A CMake file is provided for automating the build process. All you need to do is
download CMake, navigate to the build folder and build:

`cd build && cmake && make `

You can then run the application providing your image data and their labels for training.
Sample data are included in this repo, you can run it like this:

`./Main ../resources/train-images.idx3-ubyte ../resources/train-labels.idx1-ubyte`

