#include "objectrec.h"

// Custom morphological operations written from scratch
// Strategy: Opening (erode then dilate) to remove small noise,
//           then Closing (dilate then erode) to fill holes in objects

static cv::Mat erode(const cv::Mat& src, int ksize) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
    int half = ksize / 2;
    for (int r = half; r < src.rows - half; r++) {
        for (int c = half; c < src.cols - half; c++) {
            bool allWhite = true;
            for (int kr = -half; kr <= half && allWhite; kr++)
                for (int kc = -half; kc <= half && allWhite; kc++)
                    if (src.at<uchar>(r+kr, c+kc) == 0)
                        allWhite = false;
            dst.at<uchar>(r, c) = allWhite ? 255 : 0;
        }
    }
    return dst;
}

static cv::Mat dilate(const cv::Mat& src, int ksize) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
    int half = ksize / 2;
    for (int r = half; r < src.rows - half; r++) {
        for (int c = half; c < src.cols - half; c++) {
            bool anyWhite = false;
            for (int kr = -half; kr <= half && !anyWhite; kr++)
                for (int kc = -half; kc <= half && !anyWhite; kc++)
                    if (src.at<uchar>(r+kr, c+kc) == 255)
                        anyWhite = true;
            dst.at<uchar>(r, c) = anyWhite ? 255 : 0;
        }
    }
    return dst;
}

cv::Mat applyMorphology(const cv::Mat& binary) {
    // Opening: removes small noise pixels
    cv::Mat opened = dilate(erode(binary, 3), 3);
    // Closing: fills small holes inside objects
    cv::Mat closed = erode(dilate(opened, 5), 5);
    return closed;
}
