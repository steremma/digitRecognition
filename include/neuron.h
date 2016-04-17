#include <stdlib.h>

#ifndef NEURON_H
#define NEURON_H

class Neuron{

private:
	int numInputs;
	double value;
	double z;
	
public:		
	Neuron(){};
	float *weights;
	float *d_weights;
	Neuron(int n,int id);
	void initialize(int id);
	double get_value() { return value; };
	void set_value(float val) { value = val; };
	double get_z() { return z; };
	void set_z(float val) { z = val; };
};

#endif
