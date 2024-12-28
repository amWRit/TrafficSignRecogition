#include "Utils.h"
#include <filesystem>
#include <random>
#include <opencv2/ml.hpp>

// Load images from directories and store them in vectors.
void load_data(const std::string& data_dir,
    std::vector<cv::Mat>& images,
    std::vector<int>& labels) {
    std::cout << "Loading images and labels...\n";
    for (int i = 0; i < NUM_CATEGORIES; ++i) {
        std::string category_path = data_dir + "/" + std::to_string(i);
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

// Function to split data into training and testing sets
void split_data(const std::vector<cv::Mat>& images, const std::vector<int>& labels,
    std::vector<cv::Mat>& trainImages, std::vector<int>& trainLabels,
    std::vector<cv::Mat>& testImages, std::vector<int>& testLabels) {
    std::cout << "\nSplitting images and labels...\n";
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


//// Placeholder function for training the model (implement your training logic here)
//void train_model(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
//    // Implement model training logic using TensorFlow C++ API or another library.
//}

// Placeholder function for training the model
void train_model(const std::vector<cv::Mat>& images,
    const std::vector<int>& labels,
    cv::Ptr<cv::ml::KNearest>& knn) {
    // Prepare data for training
    std::cout << "\nTraining the model...\n";
    cv::Mat trainData;
    cv::Mat labelsMat = cv::Mat(labels).reshape(1, labels.size()); // Convert labels to Mat

    // Flatten images into a single row
    for (const auto& img : images) {
        cv::Mat flatImg = img.reshape(1, 1); // Flatten to 1D
        trainData.push_back(flatImg);
    }

    // Convert to float if necessary
    if (trainData.type() != CV_32F) {
        trainData.convertTo(trainData, CV_32F);
    }

    // Check if trainData and labelsMat are properly formed
    if (trainData.empty() || labelsMat.empty()) {
        std::cerr << "Training data or labels are empty!" << std::endl;
        return;
    }

    std::cout << "Train Data Rows: " << trainData.rows << ", Labels Rows: " << labelsMat.rows << std::endl;

    // Train k-NN model
    knn->setDefaultK(3); // Set number of neighbors
    knn->train(trainData, cv::ml::ROW_SAMPLE, labelsMat);

    // Save the trained model if needed
    /*try {
        knn->save("knn_model.xml");
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }*/
}
// Placeholder function for evaluating the model (implement your evaluation logic here)
void evaluate_model(const cv::Ptr<cv::ml::KNearest>& knn,
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

    // Convert to float if necessary
    if (testData.type() != CV_32F) {
        testData.convertTo(testData, CV_32F);
    }

    // Check if testData is empty
    if (testData.empty()) {
        std::cerr << "Test data is empty!" << std::endl;
        return;
    }

    // Predict labels for the test data
    cv::Mat predictedLabels;
    knn->findNearest(testData, knn->getDefaultK(), predictedLabels);

    // Calculate accuracy
    int correctCount = 0;
    for (size_t i = 0; i < predictedLabels.rows; ++i) {
        if (predictedLabels.at<float>(i, 0) == testLabels[i]) {
            correctCount++;
        }
    }

    double accuracy = static_cast<double>(correctCount) / testLabels.size() * 100.0;

    // Display results
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}


// Function to predict the traffic sign category
int predict_traffic_sign(const cv::Ptr<cv::ml::KNearest>& knn, const cv::Mat& img) {
    cv::Mat flatImg = img.reshape(1, 1); // Ensure it is a single row

    // Make prediction using k-NN
    cv::Mat predictedLabel;
    knn->findNearest(flatImg, knn->getDefaultK(), predictedLabel);

    return static_cast<int>(predictedLabel.at<float>(0, 0)); // Return predicted label
}
