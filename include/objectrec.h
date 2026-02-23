#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Task 1: Thresholding (written from scratch)
cv::Mat applyThreshold(const cv::Mat& src);

// Task 2: Morphological filtering (written from scratch)
cv::Mat applyMorphology(const cv::Mat& binary);
