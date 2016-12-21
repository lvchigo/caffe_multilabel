/*
 * =====================================================================================
 *
 *	filename:  face_attributes.cpp
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


#include "image_classifier.h"

 float Image_classifier::get_len(const std::vector<float>& v) {
	double sum = 0.0;
	for (int i = 0; i < v.size(); i ++) {
		sum += (double)v[i] * (double)v[i];
	}
	return std::pow(sum, 0.5);
}

float Image_classifier::get_dist(const std::vector<float>& v1, const std::vector<float>& v2) {
	if (v1.size() != v2.size()) {
		return -1;
	}

	double sum = 0.0;
	for (int i = 0; i < v1.size(); i ++) {
		sum += ((double)v1[i] - (double)v2[i]) * ((double)v1[i] - (double)v2[i]);
	}
	return std::pow(sum, 0.5);
}

int Image_classifier::get_dist(const std::vector<long long>& v1, const std::vector<long long>& v2) {
	if (v1.size() != v2.size()) {
		return -1;
	}

	int dist = 0;
	for (int i = 0; i < v1.size(); i ++) {
		dist += __builtin_popcountll(v1[i] ^ v2[i]);
	}

	return dist;
}

Image_classifier::Image_classifier() {}

int Image_classifier::init(const string& model_file, const string& trained_file, int use_gpu, int device_id) {
	return init(model_file, "none", trained_file, use_gpu, device_id);
}

int Image_classifier::init(const string& model_file, const string& mean_file, const string& trained_file, int use_gpu, int device_id) {
	if (use_gpu == 0) {
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(device_id);
	}

	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	SetMean(mean_file);

	return 0;
}

void Image_classifier::SetMean(const string& mean_file) {
	if (mean_file == "none") {
		mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(128, 128, 128));
		return;
	}

	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_) << "Number of channels of mean file doesn't match input layer.";

	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	cv::Mat mean;
	cv::merge(channels, mean);

	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

int Image_classifier::process(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();
	//net_->ForwardPrefilled();

	return 0;
}

void Image_classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Image_classifier::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	//sample_normalized = sample_normalized / 128.0;

	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";
}

std::vector<float> Image_classifier::get_layer_features(const std::string& layer_name) {
	std::vector<float> features;

	boost::shared_ptr<Blob<float> > feature_blob = net_->blob_by_name(layer_name);
	if (NULL == feature_blob) {
		std::cout << layer_name << " feature blob null" << std::endl;
		return features;
	}

	int batch_size = feature_blob->num();
	int dim_features = feature_blob->count() / batch_size;

	const float* feature_blob_data;
	feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(0);
	std::copy(feature_blob_data, feature_blob_data + dim_features, std::back_inserter(features));

	return features;
}

std::vector<std::pair<float, int> > Image_classifier::get_result() {
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();

	std::vector<float> output(begin, end);

	std::vector<std::pair<float, int> > res;
	for (int i = 0; i < output.size(); i ++) {
		res.push_back(std::make_pair(output[i], i));
	}

	std::sort(res.rbegin(), res.rend());
	return res;
}

std::vector<float> Image_classifier::to_one(const std::vector<float>& features) {
	std::vector<float> v_one;

	float len = get_len(features);
	for (int i = 0; i < features.size(); i ++) {
		v_one.push_back(features[i] / len);
	}

	return v_one;
}

std::vector<long long> Image_classifier::to_binary(const std::vector<float>& features) {
	std::vector<long long> fs;

	long long ll = 0;
	for (int i = 0; i < features.size(); i ++) {
		int index = i % 64;

		if (features[i] > 0.5) {
			ll = ll | (1LL << index);
		}
		if (index == 63) {
			fs.push_back(ll);
			ll = 0;
		}
	}

	if (features.size() % 64 != 0) {
		fs.push_back(ll);
	}

	return fs;
}
