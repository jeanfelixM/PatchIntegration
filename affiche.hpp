#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>


void afficher_patch(const cv::Mat& patch, const cv::Mat& depths, const cv::Mat& im, const float opacity);
void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor,bool color);