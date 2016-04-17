#include "image.h"
#include "neuronLayer.h"
#include <math.h>

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

//********************** CLASS NeuralNetwork **********************
//this class models the ANN, with its number of inputs, number of outputs.....
//and the necessary methods for the creation and
class NeuralNetwork{

private:

	int numInputs;
	int numOutputs;
	int numHiddenLayers;				
	int numNeuronsPerHiddenLayer;

public:	
	NeuronLayer *vecLayer;			//[numHiddenLayers+2];		//in order for input and output level to be included
	NeuralNetwork(){};
	NeuralNetwork(int numInputs,int numOutputs,int numHiddenLayers,int numNeuronsPerHiddenLayer);
	void createNet();
	int predict(Image image);
	void forwardPropagation(Image sample);
	void sigmoid(int layer);
	double computeCost(int label, float *output);
	void softMax();
	void updateWeights();
	double* computeSoftMaxDerivative();
 	void computeOutputDelta(int label);
	void backPropagation(int label,int image_count);
	double* computeSigmoidDerivative(int layer);

  
};

#endif
