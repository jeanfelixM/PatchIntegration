#include "patch.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

//Une amélioration pourrait être d'utiliser autre chose que zncc pour déterminer la similarité entre deux patchs
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget) {
    //si les patchs ont la même taille
    if (patchsource.size() != patchtarget.size()) {
        throw std::invalid_argument("Les patchs doivent être de la même taille");
    }

    // A MODIFIER : patchsource et patchtarget sont des matrices de coordonnées, pas de valeur des pixels -> pas bon pour l'instant

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

float patch_integration(Vec2f point,float depthinit, const cv::Mat& imsource,const cv::Mat& imtarget, const::cv::Mat& K, const cv::Mat& P){
    Mat patchsource, patchtarget;
    create_patch(imsource, point, patchsource, 5);

    int N = 10;
    float depthmax = 0;
    float depth = depthinit;
    float depthstep = 0.1;
    float maxzncc = 0;
    /*centre la recherche autour de depthinit*/
    depth -= (N/2)*depthstep;
    
    Mat integratedpatch = Mat::zeros(patchsource.rows, 3);
    normalsIntegration(patchsource, integratedpatch);
    //float depthcentre = .... (prendre la depth du point central du patch)
    integratedpatch.col(2) = integratedpatch.col(2)/depthcentre

    for (int i = 1; i < N; i++){
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
            cv::Mat w2 = (pointR2.at<float>(2)).rowRange(0,2)
            cv::Mat p2 = K * (w2);

            //coordonnées pixel entieres            
            int targetX = static_cast<int>(p2.at<float>(0));
            int targetY = static_cast<int>(p2.at<float>(1));

            //le point projeté est à l'intérieur des limites de l'image
            if (targetX >= 0 && targetX < patchtarget.cols && targetY >= 0 && targetY < patchtarget.rows) {
                patchtarget.at<cv::Vec2f>(i) = Vec2f(targetX, targetY);
            }
        
    }
}

void create_patch(const cv::Mat& im, const Vec2f point, cv::Mat& patch, int size){
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
    patch = Mat::zeros(size*size,2, CV_32FC2);
    for (int i = x1; i < x2; i++){
        for (int j = y1; j < y2; j++){
            patch.at<Vec2f>((i-x1)*(j-y1)+(j-y1)) = Vec2f(i,j);
        }
    }

}

