#include <cstring>
#include <ctime>
#include <iostream>
#include "patch.hpp"
#include "normals.hpp"
#include <opencv2/core/core.hpp>

using namespace std;

int main(int argc, char** argv)
{
	
	cv::Mat normalmap;
	cv::Mat imsource;
	cv::Mat imtarget;
	cv::Mat K;
	cv::Mat P;
	//normalsEstimation(imsource, normalmap);
	load_params(imsource, imtarget, K, P, normalmap, GT_DIRS);
	cv::Mat depthmap = cv::Mat::zeros(imsource.rows, imsource.cols, CV_32F);
	float depth;
	for (int i = 0; i < im.rows;i++){
		for (int j = 0; j < im.cols;j++){
			cv::Point2f point(i, j);
			depth = patch_integration(point,depthinit, imsource, imtarget, K, P);
			depthmap.at<float>(i,j) = depth;
		}
	}

	cv::imshow("depthmap", depthmap);
}

void load_params(cv::Mat& imsource, cv::Mat& imtarget, cv::Mat& K, cv::Mat& P,cv::Mat& normalmap, std::string DIRS){
	for 
	load_normals(normalmap, filename);
}