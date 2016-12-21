#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "image_classifier.h"

int main(int argc, char** argv) {
	if (argc != 10) {
		std::cerr << "Usage: " << argv[0]
		<< " deploy.prototxt "
		<< " mean_file "
		<< " trained.caffemodel "
		<< " use_gpu "
		<< " device_id "
		<< " layer_name "
		<< " paths.csv "
		<< " out_paths.txt "
		<< " out_features.txt"
		<< std::endl;

		return 1;
	}

	std::string deploy_file = argv[1];
	std::string mean_file = argv[2];
	std::string trained_file = argv[3];
	int use_gpu = atoi(argv[4]);
	int device_id = atoi(argv[5]);

	std::string layer_name = argv[6];

	Image_classifier image_classifier;
	image_classifier.init(deploy_file, mean_file, trained_file, use_gpu, device_id);

	std::ifstream infile(argv[7]);

	std::ofstream outfile_paths(argv[8]);
	std::ofstream outfile_features(argv[9]);

	std::string value;
	int count = 0;
	while(getline(infile, value)) {
		std::string path = value;

		cv::Mat img = cv::imread(path, 1);

		if (img.rows < 32 || img.cols < 32) {
			std::cout << "image size too small: " << path << std::endl;
			continue;
		}

		image_classifier.process(img);
		std::vector<float> features = image_classifier.get_layer_features(layer_name);

		outfile_paths << path << std::endl;

		for (int i = 0; i < features.size(); i ++) {
			outfile_features << features[i] << " ";
		}
		outfile_features << std::endl;

		count ++;
		if (count % 100 == 0) {
			std::cout << "count: " << count << std::endl;
		}
	}

	infile.close();
	outfile_paths.close();
	outfile_features.close();

	std::cout << "Down!" << std::endl;
	return 0;
}