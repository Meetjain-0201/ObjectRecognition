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
        std::string fullPath = imgDir + fname;
        cv::Mat src = cv::imread(fullPath);
        if (src.empty()) {
            std::cout << "Could not load: " << fullPath << std::endl;
            continue;
        }

        cv::Mat binary = applyThreshold(src);

        cv::imshow("Original - " + fname, src);
        cv::imshow("Thresholded - " + fname, binary);

        std::string outPath = "C:/Users/meetj/Downloads/ObjectRecognition/results/thresh_" + fname;
        cv::imwrite(outPath, binary);
        std::cout << "Processed: " << fname << std::endl;

        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    std::cout << "Done. Results saved in results/" << std::endl;
    return 0;
}
