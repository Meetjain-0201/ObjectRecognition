#include "objectrec.h"

std::vector<RegionInfo> segmentRegions(const cv::Mat& binary, cv::Mat& labelViz) {
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);

    int minArea = 500;
    int imgW = binary.cols, imgH = binary.rows;

    // Random color palette
    std::vector<cv::Vec3b> colors(numLabels);
    colors[0] = cv::Vec3b(0,0,0); // background black
    for (int i = 1; i < numLabels; i++)
        colors[i] = cv::Vec3b(rand()%200+55, rand()%200+55, rand()%200+55);

    labelViz = cv::Mat::zeros(binary.size(), CV_8UC3);
    std::vector<RegionInfo> regions;

    for (int i = 1; i < numLabels; i++) {
        int area  = stats.at<int>(i, cv::CC_STAT_AREA);
        int x     = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y     = stats.at<int>(i, cv::CC_STAT_TOP);
        int w     = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h     = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Skip small regions and regions touching the boundary
        if (area < minArea) continue;
        if (x <= 1 || y <= 1 || x+w >= imgW-1 || y+h >= imgH-1) continue;

        RegionInfo r;
        r.label    = i;
        r.centroid = cv::Point2f(centroids.at<double>(i,0), centroids.at<double>(i,1));
        r.area     = area;
        r.boundingBox = cv::Rect(x, y, w, h);
        regions.push_back(r);

        // Color the region
        for (int row = 0; row < binary.rows; row++)
            for (int col = 0; col < binary.cols; col++)
                if (labels.at<int>(row,col) == i)
                    labelViz.at<cv::Vec3b>(row,col) = colors[i];
    }

    // Sort by area descending, keep largest
    std::sort(regions.begin(), regions.end(),
        [](const RegionInfo& a, const RegionInfo& b){ return a.area > b.area; });

    return regions;
}
