# Traffic Sign Recognition Using k-NN

This project implements a traffic sign recognition system using the k-Nearest Neighbors (k-NN) algorithm, leveraging the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system is designed to classify traffic signs from images, making it suitable for applications in autonomous driving and road safety.

## Features

- **Data Loading**: Efficiently loads and preprocesses images from the GTSRB dataset.
- **Model Training**: Trains a k-NN classifier on the training dataset, with support for customizable training-test splits.
- **Prediction**: Allows users to input an image path for real-time traffic sign classification.
- **Evaluation**: Computes model accuracy using a designated test dataset.

## Requirements

- OpenCV
- C++17 or later
- CMake (for building the project)

## Usage

1. Run the executable:
```./traffic_sign_recognition```

2. Uses the trained model to test on random images from gtsrb-test-data folder

## Evaluation

The model's accuracy can be evaluated using a test dataset, which can be specified during the loading process.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/)
- [Integrating OpenCV with Visual Studio C++ Projects on Windows](https://christianjmills.com/posts/opencv-visual-studio-getting-started-tutorial/windows/)
