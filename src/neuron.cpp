#include "neuron.h"
using namespace std;

Neuron::Neuron(int n,int id){
	int i;
	numInputs=n;
	if(n) {
	    weights=new float [numInputs];
	    d_weights=new float[numInputs];
	    initialize(id);
	}
}	
	
void Neuron::initialize(int id){
  
  srand(id+2); // use current time as seed for random generator
  for(int i=0;i<numInputs;i++){
    weights[i] = rand()/float(RAND_MAX);
    d_weights[i] = 0;
    weights[i] = (weights[i] - 0.5)/2;
    //cout << "im the neuron: " << id << " of the current layer and the weight index:" << i << " has this value: "<< weights[i]<<endl;
    //wait();
  }
  
}

