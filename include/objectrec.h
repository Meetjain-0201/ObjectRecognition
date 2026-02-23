#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

cv::Mat applyThreshold(const cv::Mat& src);
cv::Mat applyMorphology(const cv::Mat& binary);

struct RegionInfo {
    int label;
    cv::Point2f centroid;
    double area;
    cv::Rect boundingBox;
    float theta;
    float minE1, maxE1, minE2, maxE2;
};
std::vector<RegionInfo> segmentRegions(const cv::Mat& binary, cv::Mat& labelViz);

struct FeatureVector {
    double percentFilled;
    double hwRatio;
    double hu1, hu2, hu3;
};
FeatureVector computeFeatures(const cv::Mat& binary, const RegionInfo& region, cv::Mat& display);

struct TrainingEntry {
    std::string label;
    FeatureVector features;
};
void saveTrainingData(const std::vector<TrainingEntry>& db, const std::string& path);
std::vector<TrainingEntry> loadTrainingData(const std::string& path);
std::string classify(const FeatureVector& fv, const std::vector<TrainingEntry>& db, double threshold=3.0);

// Task 9: CNN Embeddings
void prepEmbeddingImage(const cv::Mat& frame, cv::Mat& embimage,
                         int cx, int cy, float theta,
                         float minE1, float maxE1, float minE2, float maxE2);
cv::Mat getEmbedding(const cv::Mat& roi, cv::dnn::Net& net);
double embeddingDistance(const cv::Mat& a, const cv::Mat& b);
