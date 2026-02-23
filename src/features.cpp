#include "objectrec.h"

FeatureVector computeFeatures(const cv::Mat& binary, const RegionInfo& region, cv::Mat& display) {
    FeatureVector fv;

    cv::Mat mask = cv::Mat::zeros(binary.size(), CV_8UC1);
    binary(region.boundingBox).copyTo(mask(region.boundingBox));

    cv::Moments m = cv::moments(mask, true);
    if (m.m00 == 0) { fv = {0,0,0,0,0}; return fv; }

    double mu20 = m.mu20 / m.m00;
    double mu02 = m.mu02 / m.m00;
    double mu11 = m.mu11 / m.m00;
    double theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

    double huArr[7];
    cv::HuMoments(m, huArr);
    fv.hu1 = -copysign(1.0, huArr[0]) * log10(abs(huArr[0]) + 1e-10);
    fv.hu2 = -copysign(1.0, huArr[1]) * log10(abs(huArr[1]) + 1e-10);
    fv.hu3 = -copysign(1.0, huArr[2]) * log10(abs(huArr[2]) + 1e-10);

    double bboxArea = region.boundingBox.width * region.boundingBox.height;
    fv.percentFilled = (bboxArea > 0) ? region.area / bboxArea : 0;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::RotatedRect rrect;
    if (!contours.empty()) {
        std::vector<cv::Point> allPts;
        for (auto& c : contours) allPts.insert(allPts.end(), c.begin(), c.end());
        rrect = cv::minAreaRect(allPts);
        float rw = rrect.size.width, rh = rrect.size.height;
        fv.hwRatio = (rw > 0 && rh > 0) ? (float)std::min(rw,rh) / std::max(rw,rh) : 1.0;

        // Compute axis extents for embedding
        float cosT = cos(theta), sinT = sin(theta);
        float minE1=1e9, maxE1=-1e9, minE2=1e9, maxE2=-1e9;
        int cx = (int)region.centroid.x, cy = (int)region.centroid.y;
        for (auto& c : contours) {
            for (auto& p : c) {
                float dx = p.x - cx, dy = p.y - cy;
                float e1 =  dx*cosT + dy*sinT;
                float e2 = -dx*sinT + dy*cosT;
                minE1 = std::min(minE1, e1); maxE1 = std::max(maxE1, e1);
                minE2 = std::min(minE2, e2); maxE2 = std::max(maxE2, e2);
            }
        }
        // Store into region (const_cast since we need to update)
        const_cast<RegionInfo&>(region).theta = theta;
        const_cast<RegionInfo&>(region).minE1 = minE1;
        const_cast<RegionInfo&>(region).maxE1 = maxE1;
        const_cast<RegionInfo&>(region).minE2 = minE2;
        const_cast<RegionInfo&>(region).maxE2 = maxE2;
    } else {
        fv.hwRatio = 1.0;
    }

    // Draw
    cv::cvtColor(binary, display, cv::COLOR_GRAY2BGR);
    if (!contours.empty()) {
        cv::Point2f pts[4]; rrect.points(pts);
        for (int i = 0; i < 4; i++)
            cv::line(display, pts[i], pts[(i+1)%4], cv::Scalar(0,255,0), 2);
    }
    double axisLen = std::max(region.boundingBox.width, region.boundingBox.height) / 2.0;
    cv::Point2f cx2 = region.centroid;
    cv::Point2f p1(cx2.x + axisLen*cos(theta), cx2.y + axisLen*sin(theta));
    cv::Point2f p2(cx2.x - axisLen*cos(theta), cx2.y - axisLen*sin(theta));
    cv::line(display, p1, p2, cv::Scalar(0,0,255), 2);
    cv::circle(display, cx2, 5, cv::Scalar(255,0,0), -1);
    std::string txt = "Fill:" + std::to_string(fv.percentFilled).substr(0,4);
    cv::putText(display, txt, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);

    return fv;
}
