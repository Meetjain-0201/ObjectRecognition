#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat applyThreshold(const cv::Mat& src);
cv::Mat applyMorphology(const cv::Mat& binary);

struct RegionInfo {
    int label;
    cv::Point2f centroid;
    double area;
    cv::Rect boundingBox;
};
std::vector<RegionInfo> segmentRegions(const cv::Mat& binary, cv::Mat& labelViz);
