#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

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

        if (!regions.empty()) {
            cv::Mat featDisplay;
            FeatureVector fv = computeFeatures(cleaned, regions[0], featDisplay);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << fname << " -> percentFilled=" << fv.percentFilled
                      << " hwRatio=" << fv.hwRatio
                      << " hu1=" << fv.hu1
                      << " hu2=" << fv.hu2
                      << " hu3=" << fv.hu3 << std::endl;

            cv::imshow("Original - " + fname, src);
            cv::imshow("Regions - " + fname, labelViz);
            cv::imshow("Features - " + fname, featDisplay);

            cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/regions_" + fname, labelViz);
            cv::imwrite("C:/Users/meetj/Downloads/ObjectRecognition/results/features_" + fname, featDisplay);
        }

        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 0;
}
