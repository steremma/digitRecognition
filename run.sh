export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
echo nvcc
nvcc -O4 -c parallel.cu -o parallel.o
echo g++
g++ -o parallel -L/usr/local/cuda/lib64 -lcuda -lcudart parallel.o
echo running the program
./parallel resources/train-images.idx3-ubyte resources/train-labels.idx1-ubyte

