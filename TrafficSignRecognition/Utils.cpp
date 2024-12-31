#include "Utils.h"
#include <filesystem>
#include <unordered_map>
#include <random>
#include <opencv2/ml.hpp>
#include <fstream>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>

std::unordered_map<std::string, int> test_data_map;
std::mutex mtx; // Mutex for synchronizing access to predictedLabels

//function to verify data consistency
void print_data_statistics(const cv::Mat& data, const std::string& name) {
    double minVal, maxVal;
    cv::minMaxLoc(data, &minVal, &maxVal);
    cv::Scalar mean, stddev;
    cv::meanStdDev(data, mean, stddev);

    std::cout << "\nData statistics for " << name << ":" << std::endl;
    std::cout << "Min value: " << minVal << std::endl;
    std::cout << "Max value: " << maxVal << std::endl;
    std::cout << "Mean: " << mean[0] << std::endl;
    std::cout << "StdDev: " << stddev[0] << std::endl;
    std::cout << "Shape: " << data.rows << "x" << data.cols << std::endl;
}

// Function to load images from 43 category folders (0-42) and add corresponding labels
void load_data(const std::string& data_dir,
    std::vector<cv::Mat>& images,
    std::vector<int>& labels) {
    std::cout << "Loading training images and labels...\n";
    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        // Construct the path for the new folder structure (00000 to 00042)
        std::string category_path = data_dir + "/" + std::string(5 - std::to_string(i).length(), '0') + std::to_string(i);

        for (const auto& entry : std::filesystem::directory_iterator(category_path)) {
            if (entry.path().extension() == ".ppm") { // Only load .ppm files
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    std::cerr << "Warning: Could not open or find image: " << entry.path() << std::endl;
                    continue; // Skip this image if it cannot be loaded
                }
                preprocess_image(img); // Preprocess image if needed
                images.push_back(img);
                labels.push_back(i);
            }
        }
    }
    std::cout << "Loading completed.\n";
    std::cout << "Data Rows: " << images.size() << ", Labels Rows: " << labels.size() << std::endl;
}

// Function to load test images from test-data directory and add corresponding labels from test_data_map
void load_test_data(const std::string& test_data_dir,
    std::vector<cv::Mat>& testImages,
    std::vector<int>& testLabels) {
    std::cout << "\nLoading test images and labels...\n";
    for (const auto& entry : std::filesystem::directory_iterator(test_data_dir)) {
        if (entry.path().extension() == ".ppm") { // Only load .ppm files
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not open or find image: " << entry.path() << std::endl;
                continue; // Skip this image if it cannot be loaded
            }

            // Extract the filename including the extension
            std::string filename = entry.path().filename().string(); // This includes the .ppm extension

            // Check if the filename exists in the map and retrieve the label
            auto it = test_data_map.find(filename);
            if (it != test_data_map.end()) {
                int label = it->second; // Get the label from the map
                //std::cout << filename << " : " << label << "\n";

                preprocess_image(img); // Preprocess image if needed
                testImages.push_back(img);
                testLabels.push_back(label);
            }
            else {
                //std::cerr << "Warning: No label found for file: " << filename << "\n";
            }
        }
    }
    std::cout << "Test loading completed.\n";
    std::cout << "Test Data Rows: " << testImages.size() << ", Test Labels Rows: " << testLabels.size() << std::endl;
    //std::cout << "Example: " << testImages.front() << " : " << testLabels.front() << "\n";
}

// Function to build a map containing test data image name and its classID from gtsrb-test-data-classification.csv
void build_test_data_class_ID_map(const std::string& filePath) {
    std::cout << "\nBuilding test map...\n";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    std::string line;
    // Skip header
    std::getline(file, line);
    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string fileName, classID_str;

        if (std::getline(ss, fileName, ',') && std::getline(ss, classID_str, ','))
        {
            test_data_map[fileName] = std::stoi(classID_str);
            count++;
        }

        if (count >= NUM_TEST_CASES) break;
    }
    std::cout << "Building map completed.\n";
    std::cout << "Size: " << test_data_map.size() << std::endl;
    std::cout << "Example:\n";
    
    count = 0;
    for (const auto& entry : test_data_map) {
        std::cout << entry.first << " : " << entry.second << "\n";
        count++;
        if (count >= 5) break;
    }
    file.close();
}

// Function to split data into training and testing sets
void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
    std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
    std::vector<cv::Mat>& testImages, std::vector<int>& testLabels) {
    std::cout << "\nSplitting into Train and Test images and labels...\n";
    // Create a vector of indices
    std::vector<int> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Calculate the split index
    size_t splitIndex = static_cast<size_t>(images.size() * (1 - TEST_SIZE));

    // Split into training and testing sets
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < splitIndex) {
            trainImages.push_back(images[indices[i]]);
            trainLabels.push_back(labels[indices[i]]);
        }
        else {
            testImages.push_back(images[indices[i]]);
            testLabels.push_back(labels[indices[i]]);
        }
    }
    std::cout << "Splitting completed.\n";
    std::cout << "Train Data Rows: " << trainImages.size() << ", Train Labels Rows: " << trainLabels.size() << std::endl;
    std::cout << "Test Data Rows: " << testImages.size() << ", Test Labels Rows: " << testLabels.size() << std::endl;

}


// Preprocess image (resize and normalize if necessary)
void preprocess_image(cv::Mat& img) {
    // Convert to grayscale if it's a color image
    if (img.channels() == 3) { // Check if the image has 3 channels (BGR)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // Convert to grayscale
    }
    else if (img.channels() != 1) { // If not grayscale and not empty, handle unexpected cases
        std::cerr << "Warning: Unexpected number of channels in image: " << img.channels() << std::endl;
        return; // Skip processing if channels are unexpected
    }

    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    img.convertTo(img, CV_32F); // Convert to float
    img /= 255.0;

    // Apply histogram equalization
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    // Add Gaussian blur to reduce noise
    cv::GaussianBlur(img, img, cv::Size(3, 3), 0);

    // Enhance edges using Sobel
    cv::Mat gradX, gradY;
    cv::Sobel(img, gradX, CV_32F, 1, 0);
    cv::Sobel(img, gradY, CV_32F, 0, 1);
    cv::addWeighted(img, 0.7, (gradX + gradY), 0.3, 0, img);

    // Ensure values are still in [0,1] range
    cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);

    // print_data_statistics(img, "After preprocessing");

    // Flatten the image
    img = img.reshape(1, 1);

}


// function for training the model
void train_model(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels,
    cv::Ptr<cv::ml::SVM>& svm) {
    // Prepare data for training
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\nTraining the model...\n";
    cv::Mat trainData;
    cv::Mat labelsMat = cv::Mat(labels).reshape(1, labels.size()); // Convert labels to Mat

    // Flatten images into a single row
    for (const auto& img : images) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        trainData.push_back(flatImg);
    }

    // print_data_statistics(trainData, "Training data");

    // Normalize pixel values
    //trainData /= 255.0; // Normalize pixel values to [0, 1]

    // Convert to float if necessary
    if (trainData.type() != CV_32F) {
        trainData.convertTo(trainData, CV_32F);
    }

    // Check if trainData and labelsMat are properly formed
    if (trainData.empty() || labelsMat.empty()) {
        std::cerr << "Training data or labels are empty!" << std::endl;
        return;
    }

    std::cout << "Train Data Rows: " << trainData.rows << ", Train Labels Rows: " << labelsMat.rows << std::endl;

    // Create and configure SVM
    svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);

    // Find optimal parameters
    double bestAccuracy;
    double bestC = 100, bestGamma = 0.01;
    //auto [bestC, bestGamma] = optimize_svm_parameters(trainData, labelsMat, bestAccuracy);

    // Train final model with best parameters
    svm->setC(bestC);
    svm->setGamma(bestGamma);
    svm->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;
}

// Function for evaluating the trained model by testing against test-data
void evaluate_model(const cv::Ptr<cv::ml::SVM>& svm,
    const std::vector<cv::Mat>& testImages,
    const std::vector<int>& testLabels) {
    // Prepare data for evaluation
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\nEvaluating the model (splitted train vs test data)...\n";
    cv::Mat testData;

    // Flatten test images into a single row
    for (const auto& img : testImages) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        testData.push_back(flatImg);
    }

    // Normalize pixel values
    //testData /= 255.0; // Normalize pixel values to [0, 1]

    // Convert to float if necessary
    if (testData.type() != CV_32F) {
        testData.convertTo(testData, CV_32F);
    }

    // Check if testData is empty
    if (testData.empty()) {
        std::cerr << "Test data is empty!" << std::endl;
        return;
    }
    
    // == WITHOUT THREADING ==
    // Predict labels for the test data using multiple threads
    //cv::Mat predictedLabels;
    //knn->findNearest(testData, knn->getDefaultK(), predictedLabels);

    //// Calculate accuracy
    //int correctCount = 0;
    //for (size_t i = 0; i < predictedLabels.rows; ++i) {
    //    // std::cout << predictedLabels.at<float>(i, 0) << " : " << testLabels[i] << "\n";
    //    if (predictedLabels.at<float>(i, 0) == testLabels[i]) {
    //        correctCount++;
    //    }
    //}
    //// == WITHOUT THREADING ==
  
    // == WITH THREADING ==
    // Number of threads to use
    const int numThreads = std::thread::hardware_concurrency(); // Get number of hardware threads
    const size_t totalImages = testImages.size();

    // Create a vector to hold futures for thread results
    std::vector<std::future<void>> futures(numThreads);
    int correctCount = 0;

    // Lambda function for processing images in parallel
    auto process_images = [&](size_t start, size_t end, int threadIndex) {
        std::cout << "Starting thread..." << threadIndex << "\n";
        cv::Mat localPredictedLabels;
        //knn->findNearest(testData.rowRange(start, end), knn->getDefaultK(), localPredictedLabels);
        svm->predict(testData.rowRange(start, end), localPredictedLabels);

        // Store results in a shared vector or process them directly here
        for (size_t i = start; i < end; ++i) {
            if (localPredictedLabels.at<float>(i - start, 0) == testLabels[i]) {
                // Increment correct count or store results as needed
                correctCount++;
            }
        }
    };

    // Divide work among threads
    size_t chunkSize = totalImages / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        size_t start = i * chunkSize;
        size_t end = (i == numThreads - 1) ? totalImages : start + chunkSize; // Handle last chunk

        futures[i] = std::async(std::launch::async, process_images, start, end, i);
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.get();
    }
    // == WITH THREADING ==

    double accuracy = static_cast<double>(correctCount) / testLabels.size() * 100.0;
    // Display results
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;
}


// Function to predict the traffic sign category
int predict_traffic_sign(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& img) {
    cv::Mat processedImg = img.clone(); // Clone to avoid modifying 
    
    // Ensure proper type and shape
    if (processedImg.type() != CV_32F) {
        processedImg.convertTo(processedImg, CV_32F);
    }
    // print_data_statistics(processedImg, "Training data");

    // Make prediction
    cv::Mat predictedLabel;
    //knn->findNearest(flatImg, knn->getDefaultK(), predictedLabel);
    svm->predict(processedImg, predictedLabel);
    return static_cast<int>(predictedLabel.at<float>(0, 0)); // Return predicted label
}

// Function to make predictions for test-images using predict_traffic_sign method
void make_predictions_on_test_set(const std::string& test_data_dir, int count, const cv::Ptr<cv::ml::SVM>& svm) {
    std::cout << "\nMaking predictions on test_set...\n";
    // Create a vector of filenames from the test_data_map
    std::vector<std::string> filenames;
    for (const auto& pair : test_data_map) {
        filenames.push_back(pair.first); // Add filename to the vector
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filenames.begin(), filenames.end(), g);

    int correct = 0;
    auto it = filenames.begin();
    for (int i = 0; i < count && it != filenames.end(); ++i, ++it) {
        std::string filePath = test_data_dir + "/" + *it;
        //std::cout << "File: " << filePath;
        cv::Mat img = cv::imread(filePath);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
        }
        preprocess_image(img);
        int predicted_label = predict_traffic_sign(svm, img);
        int actual_label = test_data_map[*it];

        if (predicted_label == -1) {
            std::cerr << "Error: Prediction failed!" << std::endl;
        }
        if (predicted_label == actual_label) {
            correct++;
        }

        // std::cout << "Predicted Traffic Sign Category: " << predicted_label << " || Actual label: " << test_data_map[*it] << std::endl;
    }
    double accuracy = (static_cast<double>(correct) / count) * 100;
    std::cout << "Accuracy on test set: " << accuracy << "%" << std::endl;
    std::cout << "Predictions completed.\n\n";
}

void make_predictions_on_loaded_set(const std::vector<cv::Mat>& images,
                                    const std::vector<int>& labels, int count,
                                    cv::Ptr<cv::ml::SVM>& svm) {
    std::cout << "\Making predictions on random samples from loaded set...\n";
    // Create a vector of indices
    std::vector<int> indices(images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (int i = 0; i < count; ++i) {

        cv::Mat img = images.at(indices[i]);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
        }

        // Predict the traffic sign category
        int predicted_label = predict_traffic_sign(svm, img);

        if (predicted_label == -1) {
            std::cerr << "Error: Prediction failed!" << std::endl;
        }

        std::cout << "Predicted Traffic Sign Category: " << predicted_label << " || Actual label: " << labels[indices[i]] << std::endl;
    }
    std::cout << "Predictions completed.\n";
}

void make_predictions_on_test_cases(const std::string& images_dir, cv::Ptr<cv::ml::SVM>& svm) {
    std::cout << "\nMaking predictions on test cases...\n";
    std::string images_path = images_dir + "/";
    for (const auto& entry : std::filesystem::directory_iterator(images_path)) {
        if (entry.path().extension() == ".ppm") { // Only load .ppm files
            std::string filePath = entry.path().string();
            cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Warning: Could not open or find image: " << entry.path() << std::endl;
                continue; // Skip this image if it cannot be loaded
            }
            preprocess_image(img); // Preprocess image if needed
            // Predict the traffic sign category
            int predicted_label = predict_traffic_sign(svm, img);

            if (predicted_label == -1) {
                std::cerr << "Error: Prediction failed!" << std::endl;
            }
            std::cout << filePath << " || Predicted Traffic Sign Category: " << predicted_label << std::endl;
        }
    }

    //std::vector<std::string> filenames = {"0", "4", "18", "25", "35"};
    //std::cout << "\nMaking predictions on test cases...\n";
    //std::vector<std::string> filenames;
    //filenames.push_back("0.ppm");
    //filenames.push_back("4.ppm");
    //filenames.push_back("18.ppm");
    //filenames.push_back("25.ppm");
    //filenames.push_back("35.ppm");
    //auto it = filenames.begin();
    //for (int i = 0; i < 5; ++i, ++it) {
    //    std::string filePath = "images/" + *it;
    //    std::cout << "File: " << filePath;
    //    cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
    //    if (img.empty()) {
    //        std::cerr << "Error: Could not open or find the image!" << std::endl;
    //    }
    //    preprocess_image(img);
    //    // Predict the traffic sign category
    //    int predicted_label = predict_traffic_sign(svm, img);

    //    if (predicted_label == -1) {
    //        std::cerr << "Error: Prediction failed!" << std::endl;
    //    }
    //    std::cout << " || Predicted Traffic Sign Category: " << predicted_label << std::endl;
    //}
    std::cout << "Predictions completed.\n";
}

// New function for parameter optimization
std::pair<double, double> optimize_svm_parameters(const cv::Mat& trainData,
    const cv::Mat& labelsMat,
    double& bestAccuracy) {

    std::cout << "\nOptimizing SVM parameters using k-fold cross validation...\n";
    auto start = std::chrono::high_resolution_clock::now();
    // Define parameter grid
    std::vector<double> gammaValues = { 0.001, 0.01, 0.1, 1.0, 10.0 };
    std::vector<double> CValues = { 0.1, 1.0, 10.0, 100.0, 1000.0 };

    double bestC = 1.0;
    double bestGamma = 0.1;
    bestAccuracy = 0;

    // Number of folds for cross-validation
    const int k_folds = 5;

    // Calculate fold size
    int fold_size = trainData.rows / k_folds;

    // Grid search with k-fold cross-validation
    for (double C : CValues) {
        for (double gamma : gammaValues) {
            double total_accuracy = 0.0;

            std::cout << "Testing C=" << C << ", gamma=" << gamma << std::endl;

            // Perform k-fold cross-validation
            for (int fold = 0; fold < k_folds; fold++) {
                // Calculate validation set range
                int validation_start = fold * fold_size;
                int validation_end = (fold == k_folds - 1) ? trainData.rows : (fold + 1) * fold_size;

                // Create training and validation sets
                cv::Mat fold_train_data, fold_train_labels;
                cv::Mat fold_val_data, fold_val_labels;

                // Split data into training and validation
                for (int i = 0; i < trainData.rows; i++) {
                    if (i >= validation_start && i < validation_end) {
                        fold_val_data.push_back(trainData.row(i));
                        fold_val_labels.push_back(labelsMat.row(i));
                    }
                    else {
                        fold_train_data.push_back(trainData.row(i));
                        fold_train_labels.push_back(labelsMat.row(i));
                    }
                }

                // Configure and train SVM for this fold
                cv::Ptr<cv::ml::SVM> fold_svm = cv::ml::SVM::create();
                fold_svm->setType(cv::ml::SVM::C_SVC);
                fold_svm->setKernel(cv::ml::SVM::RBF);
                fold_svm->setC(C);
                fold_svm->setGamma(gamma);

                // Train on fold training data
                fold_svm->train(fold_train_data, cv::ml::ROW_SAMPLE, fold_train_labels);

                // Validate on fold validation data
                cv::Mat predictions;
                fold_svm->predict(fold_val_data, predictions);

                // Calculate accuracy for this fold
                int correct = 0;
                for (int i = 0; i < predictions.rows; i++) {
                    if (predictions.at<float>(i) == fold_val_labels.at<int>(i)) {
                        correct++;
                    }
                }
                double fold_accuracy = static_cast<double>(correct) / predictions.rows;
                total_accuracy += fold_accuracy;
            }

            // Calculate average accuracy across all folds
            double avg_accuracy = total_accuracy / k_folds;
            std::cout << "Average accuracy: " << avg_accuracy * 100 << "%" << std::endl;

            // Update best parameters if necessary
            if (avg_accuracy > bestAccuracy) {
                bestAccuracy = avg_accuracy;
                bestC = C;
                bestGamma = gamma;
            }
        }
    }

    std::cout << "Best parameters found: C=" << bestC
        << ", gamma=" << bestGamma
        << ", accuracy=" << bestAccuracy * 100 << "%" << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;

    return std::make_pair(bestC, bestGamma);
}

