

#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

/********************** CLASS NeuronLayer ****************************************
	this class models a layer of the ANN. It has as private variables, the
	number of Neurons, which are contained in the layer, as well as an array
	(type Neuron) containing all those Neurons. There is also a method-constructor
	for the creation of the object
*/

class NeuronLayer{
public:
	Neuron *myNeurons;
	int numNeurons;	
	NeuronLayer(){};
	NeuronLayer(int numNeurons, int numInputsPerNeuron);
	double *delta;
};

#endif
