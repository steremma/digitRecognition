#include <stdio.h>
#include <stdlib.h> 
#include <ctime>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <math.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>

using namespace std;
 
#define NUM_OF_HIDDEN_LAYERS 2
#define bias 1
#define ACTIVATION_RESPONSE 1       //it has to be changed
#define LEARNING_RATE 0.1
#define BATCH_SIZE 1000
#define EPOCHS 1000



struct timeval startwtime, endwtime;
double seq_time;


// Template structure to pass to kernel
template <typename T>
struct KernelArray
{
    T*  _array;
    int _size;
};
 
// Function to convert device_vector to structure
template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec)
{
    KernelArray<T> kArray;
    kArray._array = thrust::raw_pointer_cast(&dVec[0]);
    kArray._size  = (int) dVec.size();
 
    return kArray;
}





struct NeuralNetwork{
    int numLayers;      
    thrust::host_vector<int> layer_size;
    thrust::host_vector<int> offset;
    thrust::host_vector<float> values;
    thrust::host_vector<float> delta;
    thrust::host_vector<float> weights;
    thrust::host_vector<float> d_weights;  
};

void rand_initialize(thrust::host_vector<float> &w)
{
  srand((unsigned)time(NULL)); // use current time as seed for random generator
  for(int i=0;i<w.size();i++){
    w[i] = rand()/float(RAND_MAX);
    w[i] = (w[i] - 0.5)/2;
  }
}
    
 
struct NeuralNetwork InitializeNeuralNetwork(vector<int> layer_s){
    
    NeuralNetwork NN;
    NN.layer_size = layer_s;
    NN.offset.resize(NN.layer_size.size()+1);
    NN.offset[0] = 0;
    NN.numLayers = layer_s.size();
    int i;
    int values_size=0; 
    int weights_size =0; 
    for (i=0;i<NN.numLayers;i++){
		if(i){
			weights_size+= NN.layer_size[i]*NN.layer_size[i-1];
		}
		NN.layer_size[i]++;
		NN.offset[i+1] = NN.offset[i] + NN.layer_size[i];
		cout <<"offset "<< NN.offset[i+1]<<endl;
		values_size += NN.layer_size[i];
    }
     
    NN.weights.resize(weights_size);
    NN.d_weights.resize(weights_size);
    NN.values.resize(values_size);
    NN.delta.resize(values_size - NN.layer_size[0]);
    for (i=0;i<NN.numLayers;i++){
       NN.values[NN.offset[i+1]-1] = -1; 
    }
 
    rand_initialize(NN.weights);

    return NN;
}

__device__ void softMax2(KernelArray<float> values,int start) {
  int sum = 0;
  for(int i = start;i<start+10;i++) {
    values._array[i] = exp(values._array[i]);
    sum += values._array[i];
  }
  for(int i = start;i<start+10;i++) {
    values._array[i] /= sum;
  }
}
__device__ void sigmoid2(float &p) {
   p = 1/(1+ exp(-(p)));	
}

// This function propagates 1 layer forward. It should be called 1 time for each hidden layer + 1 time for the output.
__global__ void propagateLayer(KernelArray<float> values, KernelArray<float> weights,int *offset,int* layerSize,int *wOffset,int currentLayer) {
  //layerSize = [785 31 41 11]
  //offset = [0 785 816 857 868]
  //wOffset = [0 785*30 785*30+31*40]
  //currentLayer = 1 || 2 || 3
  //m = threadIdx.x*currentLayer + wOffset[thisLayer-1]
  //offset will be 784, 784 + 30 , 784+30.

  int m = threadIdx.x * layerSize[currentLayer-1] + wOffset[currentLayer - 1];
  int i = offset[currentLayer] + threadIdx.x + blockIdx.x * offset[NUM_OF_HIDDEN_LAYERS + 2];
  values._array[i] = 0;
  int start_j = blockIdx.x * offset[NUM_OF_HIDDEN_LAYERS + 2] + offset[currentLayer-1];

  for(int j = start_j;j<start_j + layerSize[currentLayer-1];j++){
	values._array[i] += weights._array[m] * values._array[j];
	m ++;  
  }

  __syncthreads();
  if(currentLayer==NUM_OF_HIDDEN_LAYERS+1 && threadIdx.x == 0 ) {
  float sum = 0;

  int start = blockIdx.x * offset[NUM_OF_HIDDEN_LAYERS + 2] + offset[NUM_OF_HIDDEN_LAYERS+1];
  	for(int i = start;i<start+10;i++) {
    		values._array[i] = exp(values._array[i]);
    		sum += values._array[i];
  	}
  for(int i = start;i<start+10;i++) {
    values._array[i] /= sum;
  }
 //values._array[860] += 100 ;

  }
  else if (currentLayer<NUM_OF_HIDDEN_LAYERS+1) {
	values._array[i] = 1/(1+ exp(-(values._array[i])));    
  }
  //if (i == 860) values._array[860] = 5;
  __syncthreads();
}





int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

thrust::host_vector<float> read_image(string image_path)
{
    ifstream file (image_path.c_str());
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
	int size = n_cols*n_rows;
	
	thrust::host_vector<float> dataSet(number_of_images * size);
	unsigned char temp;
	int value;

        for(int i=0;i<number_of_images*size;++i)
        {
	    temp = 0;
	    file.read((char*)&temp,sizeof(temp));
	    value = (int)temp;
            dataSet[i] =  (value/127.5 - 1);
        }
        return dataSet;
    }
}

int* read_label(string label_path){
  ifstream file (label_path.c_str());
    if (file.is_open())
    {
	int magic_number=0;
	int number_of_labels=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
	file.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);
	int* labels = new int[number_of_labels];
	unsigned char temp;
        for(int i=0;i<number_of_labels;++i)
        {
	    file.read((char*)&temp,sizeof(temp));
	    labels[i] = (int)temp;
        }
	return labels;
    }
}


__global__ void computeOutputDelta2(float* delta,const float* values, int label,const int* offset){
  
  int off_start = offset[NUM_OF_HIDDEN_LAYERS + 1];
  int i_v = off_start + threadIdx.x + blockIdx.x * offset[NUM_OF_HIDDEN_LAYERS + 2];
  int i_d = off_start - offset[1]+ threadIdx.x + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] - offset[1]) ; //block * 868 - (block + 1) * 785
  delta[i_d] = values[i_v] - (label == threadIdx.x);
  
}


/* 
 * arxika to values exei = BATCH_SIZE arxika images
 * kai tha epistrepsei = BATCH_SIZE 10aria output values ston idio pinaka.
 * , 1 weights, 
 */
void forwardPropagation( const thrust::host_vector<float> weights,
			 thrust::host_vector<float> &values, thrust::host_vector<int> offset){
  int this_layer,i;

   
  thrust::device_vector<float> device_values = values;
  thrust::device_vector<float> device_weights = weights;
  thrust::device_vector<int> device_offset = offset;
  thrust::device_vector<int> device_layerSize;
  thrust::device_vector<int> device_wOffset;


  //layerSize = [785 31 41 11] 
  thrust::host_vector<int> layerSize(offset.size() - 1);
  //wOffset = [0 785*30 785*30+31*40 785*30+31*40+41*10]
  thrust::host_vector<int> wOffset(offset.size() - 1);
  for (i=0; i<offset.size() - 1 ; i++){
      layerSize[i] = offset[i+1] - offset[i];
      
      if (!i) wOffset[i] = 0;
      else if ( i< offset.size() - 1) wOffset[i] = wOffset[i-1] + layerSize[i-1]*(layerSize[i]-1);
  }

  device_layerSize = layerSize;
  device_wOffset = wOffset;

  //for loop gia ta layers {1,2,3} apo ta {0,1,2,3}
  for (this_layer=1; this_layer<offset.size()-1;this_layer++){ 
        
	//cout << "calling propagate layer with: " << layerSize[this_layer-1] << "threads"<< endl;
        KernelArray<int> temp = convertToKernel(device_offset);
	  propagateLayer<<< BATCH_SIZE, layerSize[this_layer] - 1 >>>(convertToKernel(device_values),
		convertToKernel(device_weights), temp._array, convertToKernel(device_layerSize)._array,
		          convertToKernel(device_wOffset)._array, this_layer);
  
  }
  //cout << "d_v" << device_values.size() << endl;
  //cout << "v" << values.size() << endl;  
  values = device_values;
  
}


//kaleitai xwris to bias. Diladi 40 kai 30 fores.
__global__ void computeLayerDelta(float* delta, float* weights, float* values, int* offset, int* layerSize, int* wOffset, int currentLayer){
	
	int i_d = offset[currentLayer] - offset[1] + threadIdx.x + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] - offset[1]) ; //block * 868 - (block + 1) * 785
	int start_next_layer = offset[currentLayer+1] - offset[1] + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] - offset[1]) ; //block * 868 - (block + 1) * 785
	int i_v = offset[currentLayer] + threadIdx.x + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2]);
	delta[i_d] = 0;   
	
	int w_start = wOffset[currentLayer];
	int w_index, next_n;
	int m=0;
	for (next_n = start_next_layer;next_n<start_next_layer + layerSize[currentLayer] - 1 ; next_n++){
		w_index = w_start + m*layerSize[currentLayer] + threadIdx.x;
		delta[i_d] += weights[w_index] * delta[next_n ]; 
		m++;
	}
	float derivative = values[i_v]*(1-values[i_v]);
	
	delta[i_d] *= derivative;
	__syncthreads();
}


__global__ void computePrevLayerDWeights(float* d_w, float* delta, float* values, int* offset, int* layerSize, int* wOffset, int currentLayer ){
	
    //layerSize = [785 31 41 11] 
    //wOffset = [0 785*30 785*30+31*40 785*30+31*40+41*10]
	int i_d = offset[currentLayer] - offset[1] + threadIdx.x + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] - offset[1]) ; // block * 868 - (block + 1) * 785 ; 
	int start_prev_layer = offset[currentLayer - 1]  + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] ) ; // block * 868 
	int d_w_index = threadIdx.x * layerSize[currentLayer-1] + wOffset[currentLayer - 1] + blockIdx.x * (wOffset[NUM_OF_HIDDEN_LAYERS + 1] );// block * (785*30+31*40+41*10)
	int prev_n;

	for(prev_n = start_prev_layer; prev_n<start_prev_layer + layerSize[currentLayer - 1] - 1;prev_n++){
		d_w[d_w_index] += values[prev_n]*delta[i_d];
		d_w_index++;
	}
	int layer_bias_index = offset[currentLayer] - offset[1] + layerSize[currentLayer] - 1 + blockIdx.x * (offset[NUM_OF_HIDDEN_LAYERS + 2] - offset[1]) ; //bias
	
	d_w[d_w_index++] += delta[layer_bias_index];//bias
	__syncthreads();
	
}


__global__ void updateWeights(float* weights, float* d_weights, int weights_size){
	
	
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id<weights_size){
		int i;
		float d_w_sum = 0;
		for(i=0;i<BATCH_SIZE;i++){
			d_w_sum += d_weights[id + i * weights_size];
			d_weights[id + i * weights_size]=0;
		}
		weights[id] -= LEARNING_RATE/BATCH_SIZE*d_w_sum;
	  
	}
	__syncthreads();
}



void backPropagation(const int label,thrust::host_vector<float> &weights,const thrust::host_vector<float> values,thrust::host_vector<float> delta,
		     const thrust::host_vector<int> offset,thrust::host_vector<float> &d_w){
 
 
 //device_delta
 //device_d_w
  thrust::device_vector<float> device_values = values;
  thrust::device_vector<float> device_weights = weights;
  thrust::device_vector<float> device_delta = delta;
  thrust::device_vector<int> device_offset = offset;
  thrust::device_vector<float> device_d_weights = d_w;  
  thrust::device_vector<int> device_layerSize;
  thrust::device_vector<int> device_wOffset;

  int i,this_layer;
  //layerSize = [785 31 41 11] 
  thrust::host_vector<int> layerSize(offset.size() - 1);
  //wOffset = [0 785*30 785*30+31*40 785*30+31*40+41*10]
  thrust::host_vector<int> wOffset(offset.size() - 1);
  for (i=0; i<offset.size() - 1 ; i++){
      layerSize[i] = offset[i+1] - offset[i];
      
      if (!i) wOffset[i] = 0;
      else if ( i< offset.size() - 1) wOffset[i] = wOffset[i-1] + layerSize[i-1]*(layerSize[i]-1);
  }

  device_layerSize = layerSize;
  device_wOffset = wOffset;
 
  //layer 3
  computeOutputDelta2<<<BATCH_SIZE, 
10>>>(convertToKernel(device_delta)._array,convertToKernel(device_values)._array,label,convertToKernel(device_offset)._array);
  
  //layers {2,1} from {0,1,2,3}
  //compute delta
  for (this_layer = offset.size() - 3;this_layer>0 ; this_layer--){

	computeLayerDelta<<<BATCH_SIZE, layerSize[this_layer] - 1>>>(convertToKernel(device_delta)._array, convertToKernel(device_weights)._array, convertToKernel(device_values)._array,
															convertToKernel(device_offset)._array, convertToKernel(device_layerSize)._array, convertToKernel(device_wOffset)._array, this_layer);
  }
  
  //layers {1,2,3} from {0,1,2,3}
  //compute d_weights
  for(this_layer=1; this_layer < offset.size()-1;this_layer++){

	  computePrevLayerDWeights<<<BATCH_SIZE, layerSize[this_layer] - 1>>> (convertToKernel(device_d_weights)._array, convertToKernel(device_delta)._array, convertToKernel(device_values)._array,
												convertToKernel(device_offset)._array, convertToKernel(device_layerSize)._array, convertToKernel(device_wOffset)._array, this_layer);

  }

  //update weights
  int numOfThreads = 1000,numOfBlocks;
  numOfBlocks = (int)(weights.size() / numOfThreads) + 1;
  
  updateWeights<<<numOfBlocks, numOfThreads>>>(convertToKernel(device_weights)._array, convertToKernel(device_d_weights)._array, weights.size());
  
	
   weights = device_weights;
}
 
int predict(thrust::host_vector<float> weights, thrust::host_vector<float> values, thrust::host_vector<int> offset){
  forwardPropagation(weights,values,offset);
  int i,index;
  float max = 0;
  for(i=offset[offset.size()-2];i<offset[offset.size()-1];i++){
    if (values[i] > max){
      max = values[i];
      index = i-offset[offset.size()-2];
    }
    
  }
  return index;
  
} 
 
 
int main(int argc, char *argv[]) {
 
   
  if(argc != 5)
  { cout << "Give 'the training set image_path', 'the training set label_path' 'the test set image_path' and 'the test set label_path'\n"; exit(1); }
  string image_path = argv[1];
  string label_path = argv[2];
  string test_image_path = argv[3];
  string test_label_path = argv[4];
  thrust::host_vector<float> imageSet = read_image(image_path);
  int* labelSet = read_label(label_path);
   
  thrust::host_vector<float> test_imageSet = read_image(test_image_path);
  int* test_labelSet = read_label(test_label_path); 

  vector<int> NeuralNetwork_layers(4);
  NeuralNetwork_layers[0] = 784;
  NeuralNetwork_layers[1] = 30;
  NeuralNetwork_layers[2] = 40;
  NeuralNetwork_layers[3] = 10;

 
  NeuralNetwork NN = InitializeNeuralNetwork(NeuralNetwork_layers);
  int i,j;
  cout << "Training begins now with batch size: "<<BATCH_SIZE << " for "<<EPOCHS<<" epochs "<<endl;
  
  
  thrust::host_vector<float> host_values(NN.values.size() * BATCH_SIZE);
  thrust::host_vector<float> host_delta(NN.delta.size() * BATCH_SIZE);
  thrust::host_vector<float> host_d_weights(NN.d_weights.size() * BATCH_SIZE);

gettimeofday (&startwtime, NULL);

//Give to the biases of each layer the value -1
  for (j=0;j<BATCH_SIZE;j++){
    for (i=1;i<NN.offset.size();i++){
	host_values[NN.offset[i]-1 + j*NN.offset[NN.numLayers]] = -1;
    }
  }


  int times_to_run = EPOCHS*BATCH_SIZE,counter=0;
  i = 0;
  while(counter<times_to_run){
      counter+=BATCH_SIZE;
      i= (i +BATCH_SIZE)%60000;  
      for (j=0;j<BATCH_SIZE;j++){
	  
	   thrust::copy(imageSet.begin()+ (i+j) *784, imageSet.begin() + (i+j+1)*784, host_values.begin()+NN.offset[NN.numLayers]*j);
      }
      forwardPropagation(NN.weights, host_values, NN.offset);
	NN.values = host_values;
      backPropagation(labelSet[i],NN.weights,NN.values,host_delta,NN.offset,host_d_weights);
      if (counter%500==0) cout <<counter<<endl;
  }
  
gettimeofday (&endwtime, NULL);
seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
cout<<"The NeuralNetwork is trained for: " << EPOCHS*BATCH_SIZE << " images in: "<<seq_time <<" seconds" << endl;
  
cout<< "Evaluating the NeuralNetwork"<< endl;
  float test_sum = 0;
  for(int i=0;i<9900;i++){
      for (j=0;j<BATCH_SIZE;j++){  
	  thrust::copy(test_imageSet.begin()+ (i+j) *784, test_imageSet.begin() + (i+j+1)*784, 
host_values.begin()+NN.offset[NN.numLayers]*j);
      }
     
     int predicted_label = predict(NN.weights,host_values,NN.offset);
     if (predicted_label==test_labelSet[i]) test_sum++;
   }
   cout <<"The NeuralNetwork's accuracy is: "<<(test_sum/100)<<"%"<<endl;
   
}







