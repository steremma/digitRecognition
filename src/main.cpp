#include "neuralNetwork.h"
#include "image.h"
#include <time.h>
#include <sys/time.h>
#include "neuron.h"
#include "neuronLayer.h"
#include "utils.h"
#define NUM_OF_HIDDEN_LAYERS 2
#define BATCH_SIZE 5

using namespace std;

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char *argv[]) {
    
  if(argc != 3) {
	 cout << "Enter the training set image_path and the label_path\n"; exit(1); 
  }
  string image_path = argv[1];
  string label_path = argv[2];
  vector<Image> imageSet = utils::read_image(image_path);
  int* labelSet = utils::read_label(label_path);	  
  NeuralNetwork test(imageSet[0].get_size(),10,NUM_OF_HIDDEN_LAYERS,imageSet[0].get_size());
  
  cout << "Training begins now with batch size: "<< BATCH_SIZE << " for 20 epochs "<<endl;
  gettimeofday (&startwtime, NULL);
  
  for(int j=0;j<1000*BATCH_SIZE;j++){
    test.forwardPropagation(imageSet[j]);
    test.backPropagation(labelSet[j],j+1);
  }
  
   gettimeofday (&endwtime, NULL);
   seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
		      + endwtime.tv_sec - startwtime.tv_sec);
  
  cout<< "The neural network was trained in : "<< seq_time<< "seconds\n";

  
  // At this point the network is trained.
   for(int j=5000;j<5010;j++){
     
     int predicted_label = test.predict(imageSet[j]);
     cout << "the real label is: "<< labelSet[j] <<endl;
     cout << "and the predicted value is: " << predicted_label << endl;
     cout<<endl;
     cout<<endl;
   }
    
  
  
  
  return 0;
}
