#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>

#ifndef IMAGE_H
#define IMAGE_H

class Image{
public:
  float *image;
  int my_rows,my_cols;
  Image() {};
  Image(int n_rows,int n_cols);
  void print_image();
  int get_size(){ return my_rows*my_cols;}
  
};

#endif
