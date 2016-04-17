#include "neuralNetwork.h"
#define ACTIVATION_RESPONSE 1
#define LEARNING_RATE 0.1
#define BATCH_SIZE 5

using namespace std;


NeuralNetwork::NeuralNetwork(int numInputs,int numOutputs,int numHiddenLayers,int numNeuronsPerHiddenLayer){
	this->numInputs=numInputs;	
	this->numOutputs=numOutputs;
	this->numHiddenLayers=numHiddenLayers;
	this->numNeuronsPerHiddenLayer=numNeuronsPerHiddenLayer;
	vecLayer=new NeuronLayer[numHiddenLayers+2];
	createNet();
	
}

//creating the ANN
void NeuralNetwork::createNet(){
	int i;
	//just to fill in the first Layer
	if(this->numHiddenLayers>0){			
		vecLayer[0]=NeuronLayer(numInputs,0);	
	
	  for(i=1;i<=this->numHiddenLayers;i++){
	    /* the 2nd argument corresponds to the num of inputs each neuron will get. That is, the number of neurons of the
	       previous layer, +1 for the bias unit. 
	       REMEMBER : Even though each layer has the numNeurons variable corresponding to the actual neurons (without the bias)
	                  the table myNeurons DOES contain the bias unit as its last element.  */
		vecLayer[i]=NeuronLayer(numNeuronsPerHiddenLayer,vecLayer[i-1].numNeurons+1);		
	  } 
	  /* the previous comment applies here too. I.e each neuron of the output has 1 input for every neuron of the last hidden 
	     layer, plus 1 for the bias */
	  vecLayer[i]=NeuronLayer(numOutputs,numNeuronsPerHiddenLayer+1);
	}
	else{
		//create output layer in case the there are no hidden layers
		vecLayer[0]=NeuronLayer(numOutputs,numInputs);			
	}
		
}

void NeuralNetwork::forwardPropagation(Image sample){
  int this_layer;
  int i;
  
  for(i=0;i<sample.get_size();i++){
    /* passing pixel values to the input layer */ 
    vecLayer[0].myNeurons[i].set_value(sample.image[i]);
  }
  /* the actual neurons took their value. Now i fill the bias pseydo-neuron too */
  vecLayer[0].myNeurons[i].set_value(1);
 
  for(this_layer=1;this_layer<=numHiddenLayers+1;this_layer++)
  {
     /* IMPORTANT: Τhe following loop will not run for the bias unit. This loop calculates the z values for each neuron
                   of the current layer as a function of the weights and previous layer neuron values. The z value for the
                   bias does not need to be calculated, it is always equEal to 1. */
     for(i=0;i<vecLayer[this_layer].numNeurons;i++){
       double sum =0;
       int j=0;
       for(j=0;j<vecLayer[this_layer-1].numNeurons+1;j++){
	    
	    sum += vecLayer[this_layer].myNeurons[i].weights[j]*vecLayer[this_layer-1].myNeurons[j].get_value(); 
       }
       /* adding the bias */
       vecLayer[this_layer].myNeurons[i].set_z( sum );
     
              
    }
    if (this_layer==numHiddenLayers+1) softMax();
    else sigmoid(this_layer);
    
  } 
}

double NeuralNetwork::computeCost(int label,float *output){
  double cost;
  double sum; 
  // to be erased!
  sum=0;
  for(int i=0;i<10;i++){
    sum+= output[i];
    if(sum!=1) cout<<"We have a bug, output sum is greater than 1"<<endl;
  }
  cost=log(output[label]);
  return (-1)*cost;
  
}
/* the 2 following functions need to manipulate only the actual neuron values.
 * That is because the bias value does not come from any z input, it is by default 
 * set to 1. */

void NeuralNetwork::softMax(){
  
  double sum=0;
  double temp[10];
  for(int i=0;i<10;i++) {
    temp[i]=exp(vecLayer[numHiddenLayers+1].myNeurons[i].get_z());
    sum+=temp[i];
  }
  for(int j=0;j<10;j++) {
    vecLayer[numHiddenLayers+1].myNeurons[j].set_value(temp[j]/sum);
  }
}

void NeuralNetwork::sigmoid(int layer){
  double temp;
  for(int j=0;j<vecLayer[layer].numNeurons;j++){
    temp=(1/(1+exp(-vecLayer[layer].myNeurons[j].get_z()/ACTIVATION_RESPONSE)));
    vecLayer[layer].myNeurons[j].set_value(temp);
  }
}

double* NeuralNetwork::computeSoftMaxDerivative(){
  double temp[10];
  double sum=0;
  double *derivative=new double[10];
  
  for(int i=0;i<10;i++){
    temp[i]=exp(vecLayer[numHiddenLayers+1].myNeurons[i].get_z());
    sum+=temp[i];
  }
  for(int j=0;j<10;j++){
    derivative[j]=(sum-temp[j])/(pow(temp[j],2)+pow((sum-temp[j]),2)+2*(sum-temp[j])*temp[j]);
  }
  return derivative;
}

void NeuralNetwork::computeOutputDelta(int label){
  for(int i=0;i<10;i++){
    vecLayer[numHiddenLayers+1].delta[i]=vecLayer[numHiddenLayers+1].myNeurons[i].get_value()-(label==i);
  }
}

double* NeuralNetwork::computeSigmoidDerivative(int layer){
  // edw kati paizei na to ksanadume.
  double *derivative = new double[vecLayer[layer].numNeurons+1];
  int i;
  for (i=0;i<vecLayer[layer].numNeurons;i++){
      derivative[i] = vecLayer[layer].myNeurons[i].get_value()*(1-vecLayer[layer].myNeurons[i].get_value());
  }
  derivative[i] = 1;
  return derivative;
}

int NeuralNetwork::predict(Image image){
  forwardPropagation(image);
  int i,index;
  double max = 0;
  for(i=0;i<10;i++){
    if (vecLayer[numHiddenLayers+1].myNeurons[i].get_value() > max) {
      max = vecLayer[numHiddenLayers+1].myNeurons[i].get_value();
      index = i;
    }
    
  }
  return index;
  
}

void NeuralNetwork::backPropagation(int label,int image_count){
  computeOutputDelta(label);
  
  
  /*find the delta vectors for each layer*/
  for(int this_layer=numHiddenLayers;this_layer>0;this_layer--){
    double* derivative = computeSigmoidDerivative(this_layer);  
    for(int this_neuron=0;this_neuron<vecLayer[this_layer].numNeurons+1;this_neuron++){
		vecLayer[this_layer].delta[this_neuron]=0;
		for (int next_n=0;next_n<vecLayer[this_layer+1].numNeurons;next_n++) {
			vecLayer[this_layer].delta[this_neuron] += 
			vecLayer[this_layer+1].myNeurons[next_n].weights[this_neuron] * 
			vecLayer[this_layer+1].delta[next_n]*derivative[this_neuron];
		}
    }
  }

  /*find the updated weight vector for each layer*/
  for(int this_layer=1;this_layer<numHiddenLayers+2;this_layer++) {
      for(int this_n=0; this_n<vecLayer[this_layer].numNeurons;this_n++) {
		int prev_n;
		for (prev_n=0; prev_n<vecLayer[this_layer-1].numNeurons;prev_n++){
			vecLayer[this_layer].myNeurons[this_n].d_weights[prev_n] += 
			LEARNING_RATE*vecLayer[this_layer-1].myNeurons[prev_n].get_value() * 
			vecLayer[this_layer].delta[this_n];	  
		}
		vecLayer[this_layer].myNeurons[this_n].d_weights[prev_n] += LEARNING_RATE*vecLayer[this_layer].delta[this_n];//bias
      }     
  }
  if(!(image_count%BATCH_SIZE)) updateWeights(); 
}

void NeuralNetwork::updateWeights(){
  int i;
  for(int this_layer=1;this_layer<numHiddenLayers+2;this_layer++) {
      for(int this_n=0; this_n<vecLayer[this_layer].numNeurons;this_n++) {
		int prev_n;
		for (prev_n=0; prev_n<vecLayer[this_layer-1].numNeurons+1;prev_n++) {
		  /* Eixame ksexasei to learning rate alla kata ta alla swsto fainetai */
			vecLayer[this_layer].myNeurons[this_n].weights[prev_n] -= vecLayer[this_layer].myNeurons[this_n].d_weights[prev_n];	
			vecLayer[this_layer].myNeurons[this_n].d_weights[prev_n]=0;
		}
      }    
  }
}
