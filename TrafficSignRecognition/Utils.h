#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>

const int IMG_WIDTH = 30;
const int IMG_HEIGHT = 30;
const int NUM_CATEGORIES = 43;
const float TEST_SIZE = 0.2f; // 40% of the data will be used for testing

void load_data(const std::string& data_dir,
	std::vector<cv::Mat>& images,
	std::vector<int>& labels);

void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
	std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
	std::vector<cv::Mat>& testImages, std::vector<int>& testLabels);

void preprocess_image(cv::Mat& img);

void train_model(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels,
	cv::Ptr<cv::ml::KNearest>& knn);

void evaluate_model(const cv::Ptr<cv::ml::KNearest>& knn,
	const std::vector<cv::Mat>& testImages,
	const std::vector<int>& testLabels);

int predict_traffic_sign(const cv::Ptr<cv::ml::KNearest>& knn,
	const cv::Mat& img);

#endif // UTILS_H
