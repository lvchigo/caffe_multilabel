/*
 * =====================================================================================
 *
 *	filename:  face_attributes.h
 *
 *	description:  face attributes interface
 *
 *	version:  1.0
 *	created:  2016-08-15
 *	revision:  none
 *	compiler:  g++
 *
 *	author:  xiaotian
 *	company:  in66.com
 *
 *	copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */

#ifndef 	IMAGE_CLASSIFIER
#define IMAGE_CLASSIFIER

#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <caffe/caffe.hpp>

using namespace caffe;

class Image_classifier {
public:
	Image_classifier();
	int init(const string& model_file, const string& trained_file, int use_gpu, int device_id);
	int init(const string& model_file, const string& mean_file, const string& trained_file, int use_gpu, int device_id);

	int process(const cv::Mat& img);

	std::vector<float> get_layer_features(const std::string& layer_name);
	std::vector<std::pair<float, int> > get_result();

	//get the length of the vector
	float get_len(const std::vector<float>&);
	//get the Euclidean distance of two vectors
	float get_dist(const std::vector<float>&, const std::vector<float>&);
	//get the hamming distance of two vectors
	int get_dist(const std::vector<long long>&, const std::vector<long long>&);

	std::vector<float> to_one(const std::vector<float>&);
	std::vector<long long> to_binary(const std::vector<float>&);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;

	void SetMean(const string& mean_file);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
};

#endif