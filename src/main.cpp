#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::string imgDir = "C:/Users/meetj/Downloads/ObjectRecognition/data/test_images/";
    std::vector<std::string> images = {"img1p3.png","img2P3.png","img3P3.png","img4P3.png","img5P3.png"};

    for (auto& fname : images) {
        cv::Mat src = cv::imread(imgDir + fname);
        if (src.empty()) { std::cout << "Could not load: " << fname << std::endl; continue; }

        cv::Mat binary  = applyThreshold(src);
        cv::Mat cleaned = applyMorphology(binary);
        cv::Mat labelViz;
        std::vector<RegionInfo> regions = segmentRegions(cleaned, labelViz);

        cv::imshow("Original - " + fname, src);
        cv::imshow("Cleaned - " + fname, cleaned);
        cv::imshow("Regions - " + fname, labelViz);

        cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/thresh_" + fname, binary);
        cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/cleaned_" + fname, cleaned);
        cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/regions_" + fname, labelViz);

        std::cout << "Processed: " << fname << " | Regions found: " << regions.size() << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
