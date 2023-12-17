#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>

/**
* Estimation des normales à partir d'une image
* @param[in] rgbimage Image RGB
* @param[out] normalmap Carte des normales
*/
void normalsEstimation(cv::Mat& rgbimage, cv::Mat& normalmap);

/**
* Integration des normales pour obtenir une carte de profondeur
* @param[in] normalmap Carte des normales
* @param[out] depthmap Carte de profondeur
*/
void normalsIntegration(cv::Mat& normalmap, cv::Mat& depthmap);