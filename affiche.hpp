#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>


void afficher_patch(const cv::Mat& patch, const cv::Mat& depths, const cv::Mat& im, const float opacity);