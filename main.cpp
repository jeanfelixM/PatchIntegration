#include <cstring>
#include <ctime>
#include <iostream>
#include "patch.hpp"
#include "normals.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	
	cv::Mat normalmap;
	cv::Mat depthmapGT;
	cv::Mat imsource;
	cv::Mat imtarget;
	cv::Mat K;
	cv::Mat P;
	//normalsEstimation(imsource, normalmap);
	if(!parseArgs(argc, argv, imsource, imtarget, normalmap, K, P, depthmapGT)) {
		cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
	}
	cv::Mat depthmap = cv::Mat::zeros(imsource.rows, imsource.cols, CV_32F);
	float depth;
	for (int i = 0; i < imsource.rows;i++){
		for (int j = 0; j < imsource.cols;j++){
			cv::Point2f point(i, j);
			//depthinit à initialiser intelligement
			depth = patch_integration(point, imsource, normalmap, imtarget, 1, K, P,true,depthmapGT);
			depthmap.at<float>(i,j) = depth;
		}
	}

	cv::imshow("depthmap", depthmap);
}

void help(const char* programName)
{
    cout << "Depth map estimation from a source and target image using normal maps" << endl
         << "Usage: " << programName << endl
         << "     -source <source image>                            # the path to the source image" << endl
         << "     -target <target image>                            # the path to the target image" << endl
         << "     -calib <calibration file>                         # the path to the XML file containing both K and P matrices" << endl
         << "     -normals <normal map image>                       # (optional) the path to the normal map image" << endl
		 << "     -depthmap <depth map image>                       # (optional) the path to the depth map image" << endl
         << endl;
}



bool parseArgs(int argc, char** argv, cv::Mat& imsource, cv::Mat& imtarget, cv::Mat normalmap, cv::Mat& depthmap, cv::Mat& K, cv::Mat& P)
{
    string sourceImagePath, targetImagePath, calibFilePath, normalMapPath,depthMapPath;
    FileStorage fs;

    if(argc < 3)
    {
        help(argv[0]);
        return false;
    }

    for(int i = 1; i < argc; i++)
    {
        const char* s = argv[i];
        if(strcmp(s, "-source") == 0)
        {
            sourceImagePath.assign(argv[++i]);
        }
        else if(strcmp(s, "-target") == 0)
        {
            targetImagePath.assign(argv[++i]);
        }
        else if(strcmp(s, "-calib") == 0)
        {
            calibFilePath.assign(argv[++i]);
        }
		else if(strcmp(s, "-normals") == 0)
        {
            normalMapPath.assign(argv[++i]);
        }
		else if(strcmp(s, "-depthmap") == 0)
        {
            depthMapPath.assign(argv[++i]);
        }
        else
        {
            cerr << "Unknown option " << s << endl;
            return false;
        }
    }

	if(calibFilePath != "") {
        fs.open(calibFilePath, FileStorage::READ);
        if(!fs.isOpened()) {
            cerr << "Could not open the calibration file" << endl;
            return false;
        }
        fs["K"] >> K;
        fs["P"] >> P;
    }
	if(!normalMapPath.empty()) {
        normalmap = imread(normalMapPath, IMREAD_COLOR);
        if(normalmap.empty()) {
            cerr << "Could not open or find the normal map image" << endl;
            return false;
        }
    }
	if (!sourceImagePath.empty()) {
		imsource = imread(sourceImagePath, IMREAD_COLOR);
		if (imsource.empty()) {
			cerr << "Could not open or find the source image" << endl;
			return false;
		}
	}
	if (!targetImagePath.empty()) {
		imtarget = imread(targetImagePath, IMREAD_COLOR);
		if (imtarget.empty()) {
			cerr << "Could not open or find the target image" << endl;
			return false;
		}
	}
	if (!depthMapPath.empty()) {
		depthmap = imread(depthMapPath, IMREAD_COLOR);
		if (depthmap.empty()) {
			cerr << "Could not open or find the depth map image" << endl;
			return false;
		}
	}
    return true;
}