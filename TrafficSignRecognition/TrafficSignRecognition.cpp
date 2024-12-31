#include <opencv2/opencv.hpp>
#include <iostream>
#include "Utils.h"

int main() {
    const std::string train_data_directory = "gtsrb-train-data"; // Fixed directory for training data
    const std::string test_data_directory = "gtsrb-test-data"; // Fixed directory for test data
    const std::string test_data_class_file = "gtsrb-test-data-classification.csv"; // csv with classication for test data
    const std::string images_dir = "images"; // Fixed directory for test images

    std::vector<cv::Mat> images, trainImages, testImages;
    std::vector<int> labels, trainLabels, testLabels;
    // Load data from the specified directory
    load_data(train_data_directory, images, labels);

    // Using gtsrb-test-data for testing instead of splitting train data
    // Accuracy was low; not used
    //build_test_data_class_ID_map(test_data_class_file);
    //load_test_data(test_data_directory, testImages, testLabels);

    // Split data into training and testing sets
    split_data(images, labels, trainImages, trainLabels, testImages, testLabels);

    // Check if any images were loaded
    if (trainImages.empty() || testImages.empty()) {
        std::cerr << "Error: No images loaded from directory!" << std::endl;
        return -1;
    }

    // Using gtsrb-test-data-classification.csv, load filename and respective classID to a map
    build_test_data_class_ID_map(test_data_class_file);

    // Create k-NN instance
    //auto knn = cv::ml::KNearest::create();
    //cv::Ptr<cv::ml::SVM> svm;
    std::cout <<"\n=== SVM MODEL ===\n";
    train_model(trainImages, trainLabels, ModelType::SVM);
    evaluate_model(testImages, testLabels, ModelType::SVM);
    make_predictions_on_test_set(test_data_directory, 5, ModelType::SVM);
    make_predictions_on_loaded_set(images, labels, 5, ModelType::SVM);
    make_predictions_on_test_cases(images_dir, ModelType::SVM);

    std::cout << "\n=== KNN MODEL ===\n";
    train_model(trainImages, trainLabels, ModelType::KNN);
    evaluate_model(testImages, testLabels, ModelType::KNN);
    make_predictions_on_test_set(test_data_directory, 5, ModelType::KNN);
    make_predictions_on_loaded_set(images, labels, 5, ModelType::KNN);
    make_predictions_on_test_cases(images_dir, ModelType::KNN);

    return 0;
}