#include "objectrec.h"
#include <opencv2/dnn.hpp>
#include <cmath>

// Attribution: prepEmbeddingImage logic adapted from Bruce Maxwell utilities.cpp
void prepEmbeddingImage(const cv::Mat& frame, cv::Mat& embimage,
                         int cx, int cy, float theta,
                         float minE1, float maxE1, float minE2, float maxE2) {
    cv::Mat rotated;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), -theta * 180.0 / M_PI, 1.0);
    int largest = (int)(1.414 * std::max(frame.cols, frame.rows));
    cv::warpAffine(frame, rotated, M, cv::Size(largest, largest));

    int left   = cx + (int)minE1;
    int top    = cy - (int)maxE2;
    int width  = (int)maxE1 - (int)minE1;
    int height = (int)maxE2 - (int)minE2;

    // Bounds check
    if (left < 0)  { width  += left;  left = 0; }
    if (top  < 0)  { height += top;   top  = 0; }
    if (left + width  >= rotated.cols) width  = rotated.cols - 1 - left;
    if (top  + height >= rotated.rows) height = rotated.rows - 1 - top;

    if (width <= 0 || height <= 0) { embimage = frame.clone(); return; }
    cv::Rect roi(left, top, width, height);
    rotated(roi).copyTo(embimage);
}

cv::Mat getEmbedding(const cv::Mat& roi, cv::dnn::Net& net) {
    const int netSize = 224;
    cv::Mat resized, blob, embedding;
    cv::resize(roi, resized, cv::Size(netSize, netSize));

    // Convert to 3 channel if grayscale
    if (resized.channels() == 1)
        cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);

    cv::dnn::blobFromImage(resized, blob,
        (1.0/255.0) * (1.0/0.226),
        cv::Size(netSize, netSize),
        cv::Scalar(124, 116, 104),
        true, false, CV_32F);

    net.setInput(blob);

    // Try to get embedding from flatten layer
    try {
        embedding = net.forward("onnx_node!resnetv22_flatten0_reshape0");
    } catch (...) {
        // Fallback: get last layer output
        embedding = net.forward();
    }

    return embedding.clone();
}

double embeddingDistance(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff = a - b;
    return cv::norm(diff, cv::NORM_L2);
}
