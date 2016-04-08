<h1> Digit Recognition using Neural Networks </h1>

This program presents a Neural Network capable of recognizing hand written digits.
This project was developed for one of my student projects namely Parallel computing. 
The goal is to create 2 version of a neural network, one that is trained serially (current version)
and one that is trained using one or more GPGPUs (written in CUDA), then prove the accerelation that HPC 
can offer

<h2> What to expect </h2>
This repo only contains the serial version which served as a benchmark for the CUDA version.
I am aware of the many design and performance flaws of the current project and do not intend to 
spend more time on it for now. However i will maintain this repo primarily for personal reasons
(this is the first serious programming project i developed as a student, i would like to remember it).

<h2> Training set </h2>

The training set was downloaded from the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> database.
It consists of 60.000 sample images and their respective labels. Each image shows a digit ranging from 0-9.

<h2> Result </h2>
The network is capable of correctly predicting more than 90% even on unknown data 
(image samples that were not used during training phase) when training with more than 20000 images.
This should take 5-10 minutes on any modern CPU.

<h2> How to use it </h2>
Unfortunately my make file has gone missing (this project was developed at 2013, 
at the time i didnt even know what Git is). I will try to upload an easy to use makefile, until then
anyone can manually compile and link everything, then supply the datasets as input args.



