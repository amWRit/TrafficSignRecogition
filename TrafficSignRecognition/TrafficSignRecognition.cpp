#include <opencv2/opencv.hpp>
#include <iostream>
#include "Utils.h"

int main() {
    const std::string train_data_directory = "gtsrb-train-data"; // Fixed directory for training data
    const std::string test_data_directory = "gtsrb-test-data"; // Fixed directory for test data
    const std::string test_data_class_file = "gtsrb-test-data-classification.csv"; // Fixed directory for test data

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

    // Create k-NN instance
    //auto knn = cv::ml::KNearest::create();

    // Train the model using the loaded images and labels
    //train_model(trainImages, trainLabels, knn);

    cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
    //train_model(trainImages, trainLabels, ann); // Train the ANN model
    try {
        // Your code that may throw an exception
        train_model(images, labels, ann); // Example function call
    }
    catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
    }
    // Evaluate model performance (optional)
    evaluate_model(ann, testImages, testLabels);

    // Using gtsrb-test-data-classification.csv, load filename and respective classID to a map
    build_test_data_class_ID_map(test_data_class_file);
    make_predictions(test_data_directory, 5, ann);
    return 0;
}