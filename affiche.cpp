#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

void afficher_patch(const cv::Mat& patch, const cv::Mat& depths, const cv::Mat& im, const float opacity){
    cv::Mat imaff = im.clone();
    cv::Mat colormap;
    cv::applyColorMap(cv::Mat(depths), colormap, cv::COLORMAP_JET);


    for (int i = 0; i < patch.rows; ++i) {
        cv::Vec3b& color = colormap.at<cv::Vec3b>(i);
        color = color * opacity + imaff.at<cv::Vec3b>(patch.at<int>(i, 0), patch.at<int>(i, 1)) * (1-opacity);
        imaff.at<cv::Vec3b>(patch.at<int>(i, 0), patch.at<int>(i, 1)) = color;
    }

    cv::imshow("Image avec patch 3D et profondeurs", imaff);
    cv::waitKey(0);

}