#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "image_classifier.h"

int main(int argc, char** argv) {
	if (argc != 9) {
		std::cerr << "Usage: " << argv[0]
		<< " deploy.prototxt "
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
	std::string trained_file = argv[2];
	int use_gpu = atoi(argv[3]);
	int device_id = atoi(argv[4]);

	std::string layer_name = argv[5];

	Image_classifier image_classifier;
	image_classifier.init(deploy_file, trained_file, use_gpu, device_id);

	std::ifstream infile(argv[6]);

	std::ofstream outfile_paths(argv[7]);
	std::ofstream outfile_features(argv[8]);

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
		std::vector<pair<float, int> > out = image_classifier.get_result();

		for (int i = 0; i < out.size(); i ++) {
			features[out[i].second] = out[i].first;
		}

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
