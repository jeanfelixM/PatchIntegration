#include "patch.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "normals.hpp"

using namespace std;
using namespace cv;

//Une amélioration pourrait être d'utiliser autre chose que zncc pour déterminer la similarité entre deux patchs
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget) {
    //si les patchs ont la même taille
    if (patchsource.size() != patchtarget.size()) {
        throw std::invalid_argument("Les patchs doivent être de la même taille");
    }
    Mat srcval,targetval;
    for (int i = 0; i < patchsource.rows; ++i) {
        srcval.push_back(imsource.at<float>(patchsource.at<Vec2i>(i).val[0], patchsource.at<Vec2i>(i).val[1]));
        targetval.push_back(imtarget.at<float>(patchtarget.at<Vec2i>(i).val[0], patchtarget.at<Vec2i>(i).val[1]));
    }

    cv::Scalar meanSource, stddevSource;
    cv::meanStdDev(patchsource, meanSource, stddevSource);
    cv::Mat normPatchSource = (patchsource - meanSource) / stddevSource;

    cv::Scalar meanTarget, stddevTarget;
    cv::meanStdDev(patchtarget, meanTarget, stddevTarget);
    cv::Mat normPatchTarget = (patchtarget - meanTarget) / stddevTarget;

    cv::Mat result;
    cv::multiply(normPatchSource, normPatchTarget, result);
    double zncc = cv::sum(result)[0];

    return zncc;
}

float patch_integration(Point2f point, const cv::Mat& imsource, const cv::Mat& normalsource,const cv::Mat& imtarget,float depthinit, const::cv::Mat& K, const cv::Mat& P){
    Mat patchsource, patchtarget, patchnormalsource;
    int size = 5;
    create_patch(imsource, point, patchsource, size,false);
    create_patch(normalsource, point, patchnormalsource, size,true);

    cv::Mat integratedpatch = Mat::zeros(patchsource.rows, 3, CV_32F);
    normalsIntegration(patchnormalsource, integratedpatch);
    float depthcentre = integratedpatch.row(size*size/2 - size/2).at<float>(2);
    integratedpatch.col(2) = integratedpatch.col(2)/depthcentre;

    int N = 10; //nombre de profondeurs testées
    float depthmax = 0;
    float depth = depthinit;
    float depthstep = 0.1;
    float maxzncc = 0;
    depth -= (N/2)*depthstep; //centre la recherche autour de depthinit
    
    //recherche ameliorable
    for(int i = 1; i < N; i++){
        depth += i*depthstep;
        source2target(depth, K, P, integratedpatch, patchtarget);
        float zncc = evaluation_patch(patchsource, patchtarget, imsource, imtarget);
        if (zncc > maxzncc){
            maxzncc = zncc;
            depthmax = depth;
        }
    }
    return depth;
}
/*
* P = (R | t)
*/
void source2target(float depth, const cv::Mat& K, const cv::Mat& P, const cv::Mat& integratedpatch, cv::Mat& patchtarget) {
    // a faire : vectoriser pour eviter les boucles
    for (int i = 0; i < integratedpatch.rows; ++i) {
            //pixel en coordonnées 3D
            
            cv::Mat pointR1 = depth * (K.inv() * integratedpatch.row(i).t());
            cv::Mat pointR2 = P * cv::Mat(cv::Vec4f(pointR1.at<float>(0), pointR1.at<float>(1), pointR1.at<float>(2), 1.0f));

            //reprojection dans I2
            cv::Mat w2 = pointR2.rowRange(0,2);
            cv::Mat p2 = K * w2;

            //coordonnées pixel entieres(peut etre pas necessaire/pas bien)         
            int targetX = static_cast<int>(p2.at<float>(0));
            int targetY = static_cast<int>(p2.at<float>(1));

            //le point projeté est à l'intérieur des limites de l'image
            if (targetX >= 0 && targetX < patchtarget.cols && targetY >= 0 && targetY < patchtarget.rows) {
                patchtarget.at<cv::Vec2f>(i) = Vec2f(targetX, targetY);
            }
        
    }
}

void create_patch(const cv::Mat& im, const Vec2f point, cv::Mat& patch, int size, bool keep_form = false){
    int x = point[0];
    int y = point[1];
    int x1 = x - size/2;
    int y1 = y - size/2;
    int x2 = x + size/2;
    int y2 = y + size/2;
    if (x1 < 0){
        x1 = 0;
        x2 = size;
    }
    if (y1 < 0){
        y1 = 0;
        y2 = size;
    }
    if (x2 > im.cols){
        x2 = im.cols;
        x1 = im.cols - size;
    }
    if (y2 > im.rows){
        y2 = im.rows;
        y1 = im.rows - size;
    }
    
    /*Creation finale du patch qui sera donc les POSITIONS dans le repère IMAGE des pixels concerné*/
    if (keep_form) {
        // Création du patch sous forme de matrice 2D (carrée)
        patch = Mat::zeros(size, size, CV_32SC2);
        for (int i = x1; i < x2; i++) {
            for (int j = y1; j < y2; j++) {
                patch.at<Vec2i>(i-x1, j-y1) = Vec2i(i, j);
            }
        }
    } 
    else {
        // Création du patch sous forme de matrice vectorisée (comme dans l'implémentation actuelle)
        patch = Mat::zeros(size*size, 2, CV_32SC1);
        for (int i = x1; i < x2; i++) {
            for (int j = y1; j < y2; j++) {
                patch.at<Vec2i>((i-x1)*size + (j-y1)) = Vec2i(i, j);
            }
        }
    }
}

