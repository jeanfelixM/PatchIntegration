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
	load_ground_truth(imsource, imtarget, K, P, normalmap, GT_DIRS);
	for (int i = 0; i < im.rows;i++){
		for (int j = 0; j < im.cols;j++){
			Vec2f point = Vec2f(i, j);
			patch_integration(point,depthinit, imsource, imtarget, K, P);
		}
	}
}

void load_ground_truth(cv::Mat& imsource, cv::Mat& imtarget, cv::Mat& K, cv::Mat& P,cv::Mat& normalmap, std::string DIRS){
	for 
	load_normals(normalmap, filename);
}