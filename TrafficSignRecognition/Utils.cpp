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

        if (count >= 1000) break;
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
    cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    img.convertTo(img, CV_32F); // Convert to float
    img = img.reshape(1, 1); // Flatten the image to a single row

}


// function for training the model
void train_model(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels,
    const cv::Ptr<cv::ml::ANN_MLP>& ann) {
    // Prepare data for training
    // auto start = std::chrono::high_resolution_clock::now();
    std::cout << "\nTraining the model...\n";
    cv::Mat trainData;
    // cv::Mat labelsMat = cv::Mat(labels).reshape(1, labels.size()); // Convert labels to Mat
    cv::Mat labelsMat = cv::Mat::zeros(labels.size(), NUM_CATEGORIES, CV_32F); // One-hot encoding

    // Flatten images into a single row
    for (const auto& img : images) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        trainData.push_back(flatImg);
    }

    // Normalize pixel values
    trainData /= 255.0; // Normalize pixel values to [0, 1]

    // Convert to float if necessary
    if (trainData.type() != CV_32F) {
        trainData.convertTo(trainData, CV_32F);
    }

    // Populate labelsMat with one-hot encoding
    for (size_t i = 0; i < labels.size(); ++i) {
        int cls_label = labels[i]; // Assuming labels are in range [0, NUM_CATEGORIES-1]
        labelsMat.at<float>(i, cls_label) = 1.0f; // Set the corresponding class index to 1
    }

    // Check if trainData and labelsMat are properly formed
    if (trainData.empty() || labelsMat.empty()) {
        std::cerr << "Training data or labels are empty!" << std::endl;
        return;
    }

    std::cout << "Train Data Rows: " << trainData.rows << ", Train Labels Rows: " << labelsMat.rows << std::endl;

    // Train k-NN model
    //knn->setDefaultK(3); // Set number of neighbors
    //knn->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);

    // Train k-NN model with different k values
    //for (int k = 1; k <= 11; k += 2) { // Test odd values from 1 to 11
    //    knn->setDefaultK(k);
    //    knn->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);
    //}

    // Save the trained model if needed
    /*try {
        knn->save("knn_model.xml");
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }*/

    // Configure the ANN architecture
    int layer_sz[] = { trainData.cols, 100, 100, NUM_CATEGORIES };
    int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
    cv::Mat layer_sizes(1, nlayers, CV_32S, layer_sz);
    ann->setLayerSizes(layer_sizes); // Example: Input layer (1024), hidden layers (128 and 64), output layer (43 classes)
    ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM); // Use sigmoid activation function
    ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.01)); // Termination criteria

    // Train the model
    ann->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);
    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Time taken: " << duration.count() / 60000000 << " minutes" << std::endl;
}
// Function for evaluating the trained model by testing against test-data
void evaluate_model(const cv::Ptr<cv::ml::ANN_MLP>& ann,
    const std::vector<cv::Mat>& testImages,
    const std::vector<int>& testLabels) {
    // Prepare data for evaluation
    std::cout << "\nEvaluating the model...\n";
    cv::Mat testData;

    // Flatten test images into a single row
    for (const auto& img : testImages) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        testData.push_back(flatImg);
    }

    // Normalize pixel values
    testData /= 255.0; // Normalize pixel values to [0, 1]

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
        std::cout << "\n Starting thread..." << threadIndex << "\n";
        cv::Mat localPredictedLabels;
        //knn->findNearest(testData.rowRange(start, end), knn->getDefaultK(), localPredictedLabels);
        ann->predict(testData.rowRange(start, end), localPredictedLabels);

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
}


// Function to predict the traffic sign category
int predict_traffic_sign(const cv::Ptr<cv::ml::ANN_MLP>& ann, const cv::Mat& img) {
    cv::Mat flatImg = img.reshape(1, 1); // Ensure it is a single row

    // Make prediction using k-NN
    cv::Mat predictedLabel;
    //knn->findNearest(flatImg, knn->getDefaultK(), predictedLabel);
    ann->predict(flatImg, predictedLabel);
    return static_cast<int>(predictedLabel.at<float>(0, 0)); // Return predicted label
}

// Function to make predictions for test-images using predict_traffic_sign method
void make_predictions(const std::string& test_data_dir, int count, const cv::Ptr<cv::ml::ANN_MLP>& ann) {
    std::cout << "\nMaking predictions...\n";
    // Create a vector of filenames from the test_data_map
    std::vector<std::string> filenames;
    for (const auto& pair : test_data_map) {
        filenames.push_back(pair.first); // Add filename to the vector
    }

    // Shuffle the indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(filenames.begin(), filenames.end(), g);

    auto it = filenames.begin();
    for (int i = 0; i < count && it != filenames.end(); ++i, ++it) {
        std::string filePath = test_data_dir + "/" + *it;
        std::cout << "File: " << filePath;
        cv::Mat img = cv::imread(filePath);
        if (img.empty()) {
            std::cerr << "Error: Could not open or find the image!" << std::endl;
        }

        preprocess_image(img);

        // Predict the traffic sign category
        int predicted_label = predict_traffic_sign(ann, img);

        if (predicted_label == -1) {
            std::cerr << "Error: Prediction failed!" << std::endl;
        }

        std::cout << "\nPredicted Traffic Sign Category: " << predicted_label << " || Actual label: " << test_data_map[*it] << std::endl;
    }
    std::cout << "Predictions completed.\n";
}