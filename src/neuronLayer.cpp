#include "neuronLayer.h"



NeuronLayer::NeuronLayer(int numNeurons, int numInputsPerNeuron){
	this->numNeurons=numNeurons;
	myNeurons=new Neuron[numNeurons+1];
	delta= new double[numNeurons+1];
	delta[numNeurons] = 1; //this is used for the "biased" delta of the output layer which should be initialized to "1".
	for(int i=0;i<numNeurons+1;i++){ 
		myNeurons[i]=Neuron(numInputsPerNeuron,i);
	}
}

