#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include "EnumClass.h"

const int IMG_WIDTH = 30;
const int IMG_HEIGHT = 30;
const int NUM_CATEGORIES = 43;
const float TEST_SIZE = 0.2f; // 20% of the data will be used for testing
const int NUM_TEST_CASES = 1000;
const int NUM_PREDICT_CASES = 5;

extern std::unordered_map<std::string, int> test_data_map;

// instances of classifiers
extern cv::Ptr<cv::ml::KNearest> knn;
extern cv::Ptr<cv::ml::SVM> svm;

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

void configure_svm();
void configure_knn();

void train_model(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels,
	ModelType modelType);

void evaluate_model(const std::vector<cv::Mat>& testImages,
	const std::vector<int>& testLabels,
	ModelType modelType);

int predict_traffic_sign(const cv::Mat& img, ModelType modelType);

void make_predictions_on_test_set(const std::string& test_data_dir, int count, ModelType modelType);

void make_predictions_on_loaded_set(const std::vector<cv::Mat>& images,
	const std::vector<int>& labels, int count, ModelType modelType);

void make_predictions_on_test_cases(const std::string& images_dir, ModelType modelType);

std::pair<double, double> optimize_svm_parameters(const cv::Mat& trainData,
	const cv::Mat& labelsMat,
	double& bestAccuracy);

#endif // UTILS_H
