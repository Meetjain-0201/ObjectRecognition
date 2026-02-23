#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string imgDir = "C:/Users/meetj/Downloads/ObjectRecognition/data/test_images/";
    std::vector<std::string> images = {
        "img1p3.png", "img2P3.png", "img3P3.png", "img4P3.png", "img5P3.png"
    };

    for (auto& fname : images) {
        cv::Mat src = cv::imread(imgDir + fname);
        if (src.empty()) { std::cout << "Could not load: " << fname << std::endl; continue; }

        cv::Mat binary = applyThreshold(src);
        cv::Mat cleaned = applyMorphology(binary);

        cv::imshow("Original - " + fname, src);
        cv::imshow("Thresholded - " + fname, binary);
        cv::imshow("Cleaned - " + fname, cleaned);

        cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/thresh_" + fname, binary);
        cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/cleaned_" + fname, cleaned);
        std::cout << "Processed: " << fname << std::endl;

        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "Done." << std::endl;
    return 0;
}
