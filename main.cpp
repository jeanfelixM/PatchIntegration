#include <cstring>


#include <ctime>
#include <iostream>
#include "patch.hpp"
#include "normals.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const float FIXED_AMOUNT = 0.1;
// Display the help for the program
void help(const char* programName);

// parse the input command line arguments
bool parseArgs(int argc, char** argv, cv::Mat& imsource, cv::Mat& imtarget, cv::Mat& normalmap, cv::Mat& depthmap, cv::Mat& K, cv::Mat& P1, cv::Mat& P2);

int main(int argc, char** argv)
{
	
	cv::Mat normalmap;
	cv::Mat depthmapGT;
	cv::Mat imsource;
	cv::Mat imtarget;
	cv::Mat K;
	cv::Mat P1, P2;
	//normalsEstimation(imsource, normalmap);
	if(!parseArgs(argc, argv, imsource, imtarget, normalmap, depthmapGT, K, P1, P2)) {
		cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
	}
    cout << "parsing fini \n";
	cv::Mat depthmap = cv::Mat::zeros(imsource.rows, imsource.cols, CV_32F);
	float depth;
    float depthinit;
	for (int i = 0; i < imsource.rows;i++){
        cout << "i : " << i << std::endl;
		for (int j = 0; j < imsource.cols;j++){
            
			cv::Point2f point(i, j);
			//depthinit à initialiser intelligement (KDtree avec les points du SfM)
            depthinit = depthmapGT.at<float>(i, j) + 5;
            cout << "depthinit : " << depthinit << std::endl;
            //cout << "on va dans patch_integration \n";
			depth = patch_integration(point, imsource, normalmap, imtarget, depthinit, K, P1, P2, true, depthmapGT);
            //cout << "on est sorti de patch_integration \n"; 
			depthmap.at<float>(i,j) = depth;
            cout << "depth : " << depth << std::endl;
		}
	}

    cout << depthmap << std::endl;

    cv::normalize(depthmap, depthmap, 0, 255, cv::NORM_MINMAX);

    cout << depthmap << std::endl;

    std::string filename = "depthmap.png";

    // Save the image in PNG format
    bool isSuccess = cv::imwrite(filename, depthmap); // imwrite returns true if the image is saved successfully

    // Check for successful saving
    if (isSuccess) {
        std::cout << "depthmap is successfully saved as " << filename << std::endl;
    }
    else {
        std::cout << "Failed to save the depthmap" << std::endl;
        return 1;
    }

    return 0;

	cv::imshow("depthmap", depthmap);
    cv::waitKey(0);
}

void help(const char* programName)
{
    cout << "Depth map estimation from a source and target image using normal maps" << endl
         << "Usage: " << programName << endl
         << "     -source <source image>                            # the path to the source image" << endl
         << "     -target <target image>                            # the path to the target image" << endl
         << "     -calib <calibration file>                         # the path to the XML file containing both K and P1 and P2 matrices" << endl
         << "     -normals <normal map image>                       # (optional) the path to the normal map image" << endl
		 << "     -depthmap <depth map image>                       # (optional) the path to the depth map image" << endl
         << endl;
}




bool parseArgs(int argc, char** argv, cv::Mat& imsource, cv::Mat& imtarget, cv::Mat& normalmap, cv::Mat& depthmap, cv::Mat& K, cv::Mat& P1, cv::Mat& P2)
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
        fs["P1"] >> P1;  // Load P1
        fs["P2"] >> P2;  // Load P2
    }
	if(!normalMapPath.empty()) {
        normalmap = imread(normalMapPath,cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        //cout << normalMapPath;
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
		depthmap = imread(depthMapPath,IMREAD_UNCHANGED);
        cout << depthMapPath;
        cv::Size sz = depthmap.size();

            // Affichage de la taille
            cout << "Width: " << sz.width << ", Height: " << sz.height << std::endl << "\n";
		if (depthmap.empty()) {
			cerr << "Could not open or find the depth map image" << endl;
			return false;
		}
	}
    return true;
}