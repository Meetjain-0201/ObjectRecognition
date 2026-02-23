#include "objectrec.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
#include <cmath>

// Custom ISODATA dynamic thresholding - written from scratch
// Samples 1/16 of pixels, runs K=2 means iteration to find threshold
cv::Mat applyThreshold(const cv::Mat& src) {
    cv::Mat gray, blurred;

    // Convert to grayscale
    if (src.channels() == 3)
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else
        gray = src.clone();

    // Slight blur to reduce noise
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // Sample 1/16 of pixels randomly
    std::vector<uchar> samples;
    int step = 4; // every 4th pixel in x and y = 1/16
    for (int r = 0; r < blurred.rows; r += step)
        for (int c = 0; c < blurred.cols; c += step)
            samples.push_back(blurred.at<uchar>(r, c));

    // ISODATA: initialize two means at 1/3 and 2/3 of range
    double m1 = 85.0, m2 = 170.0;
    for (int iter = 0; iter < 20; iter++) {
        double sum1 = 0, sum2 = 0;
        int cnt1 = 0, cnt2 = 0;
        for (uchar v : samples) {
            if (std::abs(v - m1) < std::abs(v - m2)) {
                sum1 += v; cnt1++;
            } else {
                sum2 += v; cnt2++;
            }
        }
        double new_m1 = cnt1 > 0 ? sum1 / cnt1 : m1;
        double new_m2 = cnt2 > 0 ? sum2 / cnt2 : m2;
        if (std::abs(new_m1 - m1) < 0.5 && std::abs(new_m2 - m2) < 0.5) break;
        m1 = new_m1;
        m2 = new_m2;
    }

    // Threshold = midpoint between the two cluster means
    double thresh = (m1 + m2) / 2.0;

    // Apply threshold manually (from scratch - pixels below thresh are object/dark)
    cv::Mat binary(blurred.rows, blurred.cols, CV_8UC1);
    for (int r = 0; r < blurred.rows; r++)
        for (int c = 0; c < blurred.cols; c++)
            binary.at<uchar>(r, c) = (blurred.at<uchar>(r, c) < thresh) ? 255 : 0;

    return binary;
}
