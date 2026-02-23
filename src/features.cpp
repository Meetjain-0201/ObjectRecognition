#include "objectrec.h"

FeatureVector computeFeatures(const cv::Mat& binary, const RegionInfo& region, cv::Mat& display) {
    FeatureVector fv;

    // Extract region mask
    cv::Mat mask = cv::Mat::zeros(binary.size(), CV_8UC1);
    binary(region.boundingBox).copyTo(mask(region.boundingBox));

    // Compute moments from region
    cv::Moments m = cv::moments(mask, true);
    if (m.m00 == 0) { fv = {0,0,0,0,0}; return fv; }

    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;

    // Primary axis angle
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

    // Hu moments
    double huArr[7];
    cv::HuMoments(m, huArr);
    fv.hu1 = -copysign(1.0, huArr[0]) * log10(abs(huArr[0]) + 1e-10);
    fv.hu2 = -copysign(1.0, huArr[1]) * log10(abs(huArr[1]) + 1e-10);
    fv.hu3 = -copysign(1.0, huArr[2]) * log10(abs(huArr[2]) + 1e-10);

    // Percent filled
    double bboxArea = region.boundingBox.width * region.boundingBox.height;
    fv.percentFilled = (bboxArea > 0) ? region.area / bboxArea : 0;

    // Get contours for minAreaRect
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::RotatedRect rrect;
    if (!contours.empty()) {
        // Merge all contour points
        std::vector<cv::Point> allPts;
        for (auto& c : contours)
            allPts.insert(allPts.end(), c.begin(), c.end());
        rrect = cv::minAreaRect(allPts);
        float rw = rrect.size.width, rh = rrect.size.height;
        fv.hwRatio = (rw > 0 && rh > 0) ? (float)std::min(rw,rh) / std::max(rw,rh) : 1.0;
    } else {
        fv.hwRatio = 1.0;
    }

    // Draw
    cv::cvtColor(binary, display, cv::COLOR_GRAY2BGR);

    // Oriented bounding box
    if (!contours.empty()) {
        cv::Point2f pts[4];
        rrect.points(pts);
        for (int i = 0; i < 4; i++)
            cv::line(display, pts[i], pts[(i+1)%4], cv::Scalar(0,255,0), 2);
    }

    // Primary axis
    double axisLen = std::max(region.boundingBox.width, region.boundingBox.height) / 2.0;
    cv::Point2f cx = region.centroid;
    cv::Point2f p1(cx.x + axisLen * cos(theta), cx.y + axisLen * sin(theta));
    cv::Point2f p2(cx.x - axisLen * cos(theta), cx.y - axisLen * sin(theta));
    cv::line(display, p1, p2, cv::Scalar(0,0,255), 2);
    cv::circle(display, cx, 5, cv::Scalar(255,0,0), -1);

    // Feature text
    std::string txt = "Fill:" + std::to_string(fv.percentFilled).substr(0,4);
    cv::putText(display, txt, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);

    return fv;
}
