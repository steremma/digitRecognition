#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>

namespace utils {

    /*
	* Add documentation
	*/
	int reverseInt (int i);

    /*
	* Add documentation
	*/
	std::vector<imageSample> read_image(std::string image_path);

    /*
	* Add documentation
	*/
	int* read_label(std::string label_path);

} //end of namespace utils
