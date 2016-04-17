#include "utils.h"

using namespace std;

namespace utils {
	int reverseInt (int i) 
	{
		unsigned char c1, c2, c3, c4;

		c1 = i & 255;
		c2 = (i >> 8) & 255;
		c3 = (i >> 16) & 255;
		c4 = (i >> 24) & 255;

		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	}

	vector<Image> read_image(string image_path)
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
		vector<Image> dataSet(number_of_images);
		unsigned char temp;
		int value;
		    for(int i=0;i<number_of_images;++i)
		    {
			Image temp_image = Image(n_rows,n_cols);
		        for(int j=0;j<n_rows*n_cols;++j)
		        {
			  temp = 0;
			  file.read((char*)&temp,sizeof(temp));
			  value = (int)temp;
			  temp_image.image[j] = (value/127.5 - 1); // making sure input values are mapped into [-1,1]
		        }
		        dataSet[i] = temp_image;
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
} //end of namespace utils
