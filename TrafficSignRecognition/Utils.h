#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>

const int IMG_WIDTH = 30;
const int IMG_HEIGHT = 30;
const int NUM_CATEGORIES = 43;
const float TEST_SIZE = 0.4f; // 20% of the data will be used for testing

extern std::unordered_map<std::string, int> test_data_map;

void load_data(const std::string& data_dir,
	std::vector<cv::Mat>& images,
	std::vector<int>& labels);

void load_test_data(const std::string& test_data_dir,
	std::vector<cv::Mat>& testImages,
	std::vector<int>& testLabels);

void build_test_data_class_ID_map(const std::string& filePath);

void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
	std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
	std::vector<cv::Mat>& testImages, std::vector<int>& testLabels);

void preprocess_image(cv::Mat& img);

void train_model(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels,
	const cv::Ptr<cv::ml::RTrees>& rtrees);

void evaluate_model(const cv::Ptr<cv::ml::RTrees>& rtrees,
	const std::vector<cv::Mat>& testImages,
	const std::vector<int>& testLabels);

int predict_traffic_sign(const cv::Ptr<cv::ml::RTrees>& rtrees,
	const cv::Mat& img);

void make_predictions(const std::string& test_data_dir, int count, 
	const cv::Ptr<cv::ml::RTrees>& rtrees);

#endif // UTILS_H
