#include <opencv2/opencv.hpp>
#include <iostream>
#include "Utils.h"

int main() {
    const std::string data_directory = "gtsrb-small"; // Fixed directory for training data

    std::vector<cv::Mat> images;
    std::vector<int> labels;

    // Load data from the specified directory
    load_data(data_directory, images, labels);

    // Check if any images were loaded
    if (images.empty()) {
        std::cerr << "Error: No images loaded from directory!" << std::endl;
        return -1;
    }

    // Create k-NN instance
    auto knn = cv::ml::KNearest::create();

    // Train the model using the loaded images and labels
        train_model(images, labels, knn);

    // Prepare your test data
    const std::string test_data_directory = "test-data";
    std::vector<cv::Mat> testImages; // Load or prepare your test images
    std::vector<int> testLabels;      // Load or prepare your test labels
    load_data(test_data_directory, testImages, testLabels);

    // Evaluate model performance (optional)
    evaluate_model(knn, testImages, testLabels);

    // User input for prediction
    std::string test_image;
    std::cout << "Enter path to an image for prediction (e.g., images/test_image.ppm): ";
    std::cin >> test_image;

    // Load and preprocess the test image for prediction
    cv::Mat img = cv::imread(test_image);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return -1;
    }

    preprocess_image(img);

    // Predict the traffic sign category
    int predicted_label = predict_traffic_sign(knn, img);

    if (predicted_label == -1) {
        std::cerr << "Error: Prediction failed!" << std::endl;
        return -1; // Handle prediction failure
    }

    std::cout << "Predicted Traffic Sign Category: " << predicted_label << std::endl;

    return 0;
}