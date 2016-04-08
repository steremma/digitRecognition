<h1> Digit Recognition using Neural Networks </h1>

This program presents a Neural Network capable of recognizing hand written digits.
This project was developed for one of my student projects namely Parallel computing. 
The goal is to create 2 version of a neural network, one that is trained serially (current version)
and one that is trained using one or more GPGPUs (written in CUDA), then prove the accerelation that HPC 
can offer

<h2> What to expect </h2>
This branch contains the CUDA version which was benchmarked against the serial one.
I am aware of the many design and performance flaws of the current project and do not intend to 
spend more time on it for now. However i will maintain this repo primarily for personal reasons
(this is the first serious programming project i developed as a student, i would like to remember it).

<h2> Training set </h2>

The training set was downloaded from the <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> database.
It consists of 60.000 sample images and their respective labels. Each image shows a digit ranging from 0-9.

<h2> Result </h2>
The network is capable of correctly predicting more than 90% (similarily to the serial version).
However training is now accerelated. Observations when running on NVIDIA GTX 750 showed a speed up of 
more than 10x (training in a matter of seconds as opposed to 5+ minutes needed for the serial version).

<h2> How to use it </h2>
Clone the project on a new directory, make sure the script "run.sh" is executable
(for example you can run chmod u+x run.sh) and run it.


