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
bool parseArgs(int argc, char** argv, cv::Mat& imsource, cv::Mat& normalmap, cv::Mat& depthmap, cv::Mat& K, vector<cv::Mat>& Pmats,vector<cv::Mat>& targetImages);

int main(int argc, char** argv)
{
	vector<cv::Mat> targetImages;
	cv::Mat normalmap;
	cv::Mat depthmapGT;
	cv::Mat imsource;
	cv::Mat K;
	vector<cv::Mat> Pmats;
    cv::Mat P1;
    //cout << "on va dans parse \n";
	//normalsEstimation(imsource, normalmap);
	if(!parseArgs(argc, argv, imsource, normalmap, depthmapGT, K, Pmats,targetImages)) {
		cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
	}
    cout << "parsing fini \n";
    //cout << depthmapGT << std::endl;
    cout << "taille de targetImages en dehors de parse:" << targetImages.size() << std::endl;
    if (imsource.empty() || normalmap.empty() || depthmapGT.empty() || targetImages.empty()) {
        cerr << "Une ou plusieurs images n'ont pas été chargées correctement." << endl;
        return EXIT_FAILURE;
    }
    int base = 1;
    P1 = Pmats[base];
    cout << "P1 : " << P1 << std::endl;
    // Visualisation des images
    cv::namedWindow("Image Source", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image Source", imsource);

    

    cv::namedWindow("Normal Map", cv::WINDOW_AUTOSIZE);
    cv::imshow("Normal Map", normalmap);

    cv::namedWindow("Depth Map GT", cv::WINDOW_AUTOSIZE);
    cv::imshow("Depth Map GT", depthmapGT);
    cv::waitKey(0);
    cout << "Ground truth depth map size : " << depthmapGT.size() << std::endl;
    cout << "Ground truth depth map type : " << depthmapGT.type() << std::endl;

    

	cv::Mat depthmap = cv::Mat::zeros(imsource.rows, imsource.cols, CV_32F);
	float depth;
    float depthinit;
    for (size_t i = 0; i < targetImages.size(); ++i) {
        cv::Mat imtarget = targetImages[i];
        cv::Mat P2;
        if (i >= base){
            P2 = Pmats[i+1];
        }
        else{
            P2 = Pmats[i];
        }
        //cout << "P2 : " << P2 << std::endl;
        //cv::namedWindow("Image Target", cv::WINDOW_AUTOSIZE);
        //cv::imshow("Image Target", imtarget);
        //cv::waitKey(0); // Attendre que l'utilisate
        cout << "Processing target image " << (i + 1) << " / " << targetImages.size() << endl;
        //cout << "With P2 : " << P2 << endl;
        for (int i = 0; i < imsource.rows;i++){
            //cout << "i : " << i << std::endl;
            for (int j = 0; j < imsource.cols;j++){
                
                cv::Point2f point(i, j);
                //depthinit à initialiser intelligement (KDtree avec les points du SfM)
                depthinit = depthmapGT.at<float>(i, j) ;//+ 5;
                //cout << "depthinit : " << depthinit << std::endl;
                //cout << "depthinit : " << depthinit << std::endl;
                //cout << "on va dans patch_integration \n";
                depth = patch_integration(point, imsource, normalmap, imtarget, depthinit, K, P1, P2, true, depthmapGT);
                //cout << "on est sorti de patch_integration \n"; 
                depthmap.at<float>(i,j) = depth;
                //cout << "depth : " << depth << std::endl;
            }
        }
        std::string filename = "depthmap_" + std::to_string(i) + ".tiff";

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
        //cv::imshow("depthmap", depthmap);
        //cv::waitKey(0);
    }

    //cout << depthmap << std::endl;

    

    //cout << depthmap << std::endl;

    

    return 0;
}

void help(const char* programName)
{
    cout << "Depth map estimation from images using normal maps" << endl
         << "Usage: " << programName << endl
         << "     -source <source image>                            # the path to the source image (use without -dir)" << endl
         << "     -target <target image>                            # the path to the target image (use without -dir)" << endl
         << "     -dir <images directory>                           # (optional) the path to the directory containing images (use with -sourceName)" << endl
         << "     -sourceName <source image name>                   # (optional) the name of the source image in the directory (use with -dir)" << endl
         << "     -calib <calibration file>                         # the path to the XML file containing both K and P1 and P2 matrices" << endl
         << "     -normals <normal map image>                       # (optional) the path to the normal map image" << endl
         << "     -depthmap <depth map image>                       # (optional) the path to the depth map image" << endl
         << endl;
}





bool parseArgs(int argc, char** argv, cv::Mat& imsource, cv::Mat& normalmap, cv::Mat& depthmap, cv::Mat& K, vector<cv::Mat>& Pmats,vector<cv::Mat>& targetImages)
{
    cv::Mat predepthmap;
    cv::Mat imtarget;
    string sourceImagePath, targetImagePath, calibFilePath, normalMapPath,depthMapPath,sourceName;
    FileStorage fs;
    string dirPath;
    string targetName;

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
         else if(strcmp(s, "-dir") == 0)
        {
            dirPath.assign(argv[++i]);
        }
        else if(strcmp(s, "-sourceName") == 0)
        {
            sourceName.assign(argv[++i]);
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
        bool empty = false;
        int i = 1;
        while(!empty){  
            cv::Mat P;
            std::string Pname = "P" + std::to_string(i);
            fs[Pname] >> P;
            empty = P.empty();
            if(!empty){
                Pmats.push_back(P);
                cout << "on push back venant de : " << Pname << std::endl;
            }
            i++;
        }
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
        else{
            targetImages.push_back(imtarget);
            cout << "on push back venant de : " << targetImagePath << std::endl;
        }
	}
	if (!depthMapPath.empty()) {
		predepthmap = imread(depthMapPath,cv::IMREAD_UNCHANGED);
        //cout << depthMapPath;
        cv::Size sz = predepthmap.size();

            // Affichage de la taille
            cout << "Width: " << sz.width << ", Height: " << sz.height << std::endl << "\n";
		if (predepthmap.empty()) {
			cerr << "Could not open or find the depth map image" << endl;
			return false;
		}
        //cout << "predpthmap : " << predepthmap << std::endl;
        predepthmap =  predepthmap;
        predepthmap.convertTo(depthmap, CV_32F);
        
	}
    if (!dirPath.empty() && !sourceName.empty())
    {
        // Load images from directory
        vector<cv::String> filePaths;
        cv::glob(dirPath, filePaths, false);

        for (const auto& filePath : filePaths)
        {
            if (filePath.find(sourceName) != string::npos) {
                imsource = cv::imread(filePath, cv::IMREAD_COLOR);
                cout << "dans  source image on met : " << filePath << std::endl;
                if (imsource.empty()) {
                    cerr << "Could not open or find the source image" << endl;
                    return EXIT_FAILURE;
                }
            } else {
                cv::Mat img = cv::imread(filePath, cv::IMREAD_COLOR);
                if (!img.empty()) {
                    targetImages.push_back(img);
                    cout << "dans  target image on met : " << filePath << std::endl;
                }
            }
        }

        if (imsource.empty() || targetImages.empty()) {
            cerr << "Source image or target images were not loaded correctly" << endl;
            return EXIT_FAILURE;
        }
    }
    cout << "taille de targetImages :" << targetImages.size() << std::endl;
    return true;
}