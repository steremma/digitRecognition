#include "image.h"
using namespace std;

void Image::print_image(){
  int i,j;

  for (i=0;i<my_rows;i++){
    cout << endl;
    for (j=0;j<my_cols;j++){
	if(image[i*28+j]) cout << 1;    // since the new mapping has been applied this will not work.
	else cout << 0;
    }
  }
  
  }

/* constructors and other class functions are defined here */
Image::Image(int n_rows,int n_cols){
  
  my_rows = n_rows;
  my_cols = n_cols;
  image = new float [n_rows*n_cols];
  
}



