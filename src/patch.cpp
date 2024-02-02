#include "patch.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "normals.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

float findOptimalDepth(float depthinit, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, const cv::Mat& imsource, const cv::Mat& imtarget, const cv::Mat& patchsource, float depthstep, int nb_profondeur,float debugtab[2]);
void vectorize_patch(const cv::Mat& integratedPatch, cv::Mat& vectorizedPatch);
void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor=1.5,bool color = false);

//Une amélioration pourrait être d'utiliser autre chose que zncc pour déterminer la similarité entre deux patchs
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget) {
    float nn = (sqrt(patchsource.rows));
    //conditions inutiles à priori
    if (patchsource.size() != patchtarget.size() || patchsource.rows == 0 || patchsource.cols != 2) {
        throw std::invalid_argument("Les patchs doivent être de la même taille et non vides avec 2 colonnes");
    }
    if (imsource.type() != CV_8UC3 || imtarget.type() != CV_8UC3) {
        throw std::invalid_argument("Les images sources et cibles doivent être de type CV_8UC3");
    }

    std::vector<float> srcVals, targetVals;
    srcVals.reserve(patchsource.rows);
    targetVals.reserve(patchtarget.rows);

    // Récupérer les valeurs correspondantes en niveau de gris(fait à la shlag(go interpoler directement))
    for (int i = 0; i < patchsource.rows; ++i) {
        int xSrc = patchsource.at<float>(i, 0);
        int ySrc = patchsource.at<float>(i, 1);
        cv::Vec3b pixelSrc = imsource.at<cv::Vec3b>(ySrc, xSrc);
        float graySrc = 0.299 * pixelSrc[2] + 0.587 * pixelSrc[1] + 0.114 * pixelSrc[0];
        srcVals.push_back(graySrc);

        int xTarget = patchtarget.at<float>(i, 0);
        int yTarget = patchtarget.at<float>(i, 1);
        cv::Vec3b pixelTarget = imtarget.at<cv::Vec3b>(yTarget, xTarget);
        float grayTarget = 0.299 * pixelTarget[2] + 0.587 * pixelTarget[1] + 0.114 * pixelTarget[0];
        targetVals.push_back(grayTarget);
    }

    Mat srcMat = cv::Mat(srcVals).reshape(1, 1);
    Mat targetMat = cv::Mat(targetVals).reshape(1, 1);

    Scalar meanSource, stddevSource;
    meanStdDev(srcMat, meanSource, stddevSource);
    Mat normPatchSource = (srcMat - meanSource) / stddevSource;

    Scalar meanTarget, stddevTarget;
    meanStdDev(targetMat, meanTarget, stddevTarget);
    Mat normPatchTarget = (targetMat - meanTarget) / stddevTarget;

    Mat result;
    multiply(normPatchSource, normPatchTarget, result);
    
    float zncc = cv::sum(result)[0]/nn;

    return zncc;
}




//a refacto (param et init)
float patch_integration(Point2f point, const cv::Mat& imsource, const cv::Mat& normalsource,const cv::Mat& imtarget,float depthinit, const::cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, float debugtab[2],bool debug, const cv::Mat& depthmap){
    Mat patchsource, patchtarget, patchnormalsource,prepatchnormalsource,integratedpatch;
    cv::Mat preintegratedpatch = Mat::zeros(patchsource.rows, 3, CV_32F);
    int size = 11;
    create_patch(imsource, point, patchsource, size);

    if (debug){
        
        create_patch(depthmap, point, integratedpatch, size, true);
       
    }
    else{
        create_patch(normalsource, point, prepatchnormalsource, size,false);
        //passer des coordonnées aux valeurs et un-vectoriser
        cv::Mat patchnormalsource = Mat::zeros(size, size, CV_32F);
        for (int i = 0; i < prepatchnormalsource.rows; ++i) {
            int x = prepatchnormalsource.at<int>(i,0);
            int y = prepatchnormalsource.at<int>(i,1);
            patchnormalsource.at<float>(x,y) = normalsource.at<float>(x, y);
        }
        normalsIntegration(patchnormalsource, preintegratedpatch);
        vectorize_patch(preintegratedpatch, integratedpatch);
    }
    float depthcentre = (integratedpatch.at<float>(size*size/2 - size/2,2)); 
    integratedpatch.col(2) = integratedpatch.col(2)/depthcentre;


    //a faire : prunage des points qui ne sont pas dans l'image target (comparer normal du point et direction camera) (ou a integrer dans source2target)

    int nb_profondeur = 15; // Nombre de profondeurs testées
    float depthstep = 0.1; // Ou 0.01 à tester
    float optimalDepth = findOptimalDepth(depthinit, K, P1,P2, integratedpatch, imsource, imtarget, patchsource, depthstep, nb_profondeur,debugtab);


    return optimalDepth;
}

/*
* P = (R | t)
*/
void source2target(float depth, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, cv::Mat& patchtarget,int boundcols, int boundrows) {
    cv::Mat ones = cv::Mat::ones(integratedpatch.rows, 1, CV_32F);
    cv::Mat integratedpatchHomogeneous;
    cv::hconcat(integratedpatch.colRange(0, 2), ones, integratedpatchHomogeneous);

    cv::Mat w1 = K.inv() * integratedpatchHomogeneous.t();
    

    Mat depthVec = depth * (integratedpatch.col(2)).t(); // Vecteur de profondeur

    Mat pointR1 = cv::Mat::zeros(3, integratedpatch.rows, CV_32F);
    multiply(depthVec, w1.row(0), pointR1.row(0));
    multiply(depthVec, w1.row(1), pointR1.row(1));
    multiply(depthVec, w1.row(2), pointR1.row(2));

    Mat pointR1Homogeneous;
    vconcat(pointR1, ones.t(), pointR1Homogeneous);
    Mat pointWorld = P1 * pointR1Homogeneous;
    Mat R2 = P2.rowRange(0, 3).colRange(0, 3);
    Mat t2 = P2.rowRange(0, 3).col(3);
    Mat P2rev;
    hconcat(R2.t(), (-R2.t()*t2),P2rev);
    vconcat(pointWorld, ones.t(), pointWorld);
    Mat pointR2 = P2rev * pointWorld;


    // Projetter les points dans le système de coordonnées de la caméra 2
    Mat pp2 = K * pointR2;
    divide(pp2.row(0), pp2.row(2), pp2.row(0)); // Normaliser par la troisième composante
    divide(pp2.row(1), pp2.row(2), pp2.row(1));
    Mat p2 = pp2.rowRange(0, 2).t();


    patchtarget = Mat::zeros(integratedpatch.rows, 2, CV_32F);
    for (int i = 0; i < integratedpatch.rows; ++i) {
        int targetX = std::round(p2.at<float>(i, 0));
        int targetY = std::round(p2.at<float>(i, 1));

        if (targetX >= 0 && targetX < boundcols && targetY >= 0 && targetY < boundrows) {
            patchtarget.at<float>(i, 0) = targetX;
            patchtarget.at<float>(i, 1) = targetY;
        }
    }
}


void create_patch(const cv::Mat& im, const cv::Vec2f point, cv::Mat& patch, int size, bool keepval){
    int x = point[0];
    int y = point[1];
    int x1 = x - size/2;
    int y1 = y - size/2;
    int x2 = x + size/2;
    int y2 = y + size/2;
    if (x1 < 0){
        x1 = 0;
        //x2 = size-1;
    }
    if (y1 < 0){
        y1 = 0;
        //y2 = size-1;
    }
    if (x2 >= im.cols){
        x2 = im.cols-1;
        //x1 = im.cols - size ;
    }
    if (y2 >= im.rows){
        y2 = im.rows-1;
        //y1 = im.rows - size;
    }
    /*Creation finale du patch qui sera donc les POSITIONS dans le repère IMAGE des pixels concerné*/
    int idx =0;
    float scale = 5.0;
    if (keepval){

        patch = Mat::zeros(size*size,3, CV_32F);
        for (int i = x1; (i <= x2); i++){
            for (int j = y1; j <= y2; j++){
                idx = (i-x1)*(size) +(j-y1);
                patch.at<float>(idx,0) = i;
                patch.at<float>(idx,1) = j;
                patch.at<float>(idx,2) = (im.at<float>(i,j));
            }
        }
    }
    else{   
        patch = Mat::zeros(size*size,2, CV_32F);
        for (int i = x1; (i <= x2); i++){
            for (int j = y1; j <= y2; j++){
                idx = (i-x1)*(size)+(j-y1);
                patch.at<float>(idx,0) = i;
                patch.at<float>(idx,1) = j;
            }
        }
    }
}

//pas testé
void vectorize_patch(const cv::Mat& integratedPatch, cv::Mat& vectorizedPatch) {
    int size = integratedPatch.rows; 
    vectorizedPatch = Mat::zeros(size*size, 3, CV_32FC1); 

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            float pixelValue = integratedPatch.at<float>(x, y);
            vectorizedPatch.at<Vec3f>(x*size + y) = Vec3f(static_cast<float>(x), static_cast<float>(y), pixelValue);
        }
    }
}

float findOptimalDepth(float depthinit, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, const cv::Mat& imsource, const cv::Mat& imtarget, const cv::Mat& patchsource, float depthstep, int nb_profondeur,float debugtab[2]) {
    float depthmax = 0;
    float maxzncc =-std::numeric_limits<double>::infinity();
    float depth = (depthinit - (nb_profondeur / 2) * depthstep);
    cv::Mat patchmax;
    for(int i = 0; i < nb_profondeur; i++) {
        cv::Mat patchtarget;
        source2target(depth, K, P1,P2, integratedpatch, patchtarget,imtarget.cols, imtarget.rows);
        //debugPatch(patchsource, patchtarget, imsource, imtarget,5.5);
        if (patchtarget.rows == 0) {
            continue;
        }
        
        // Ajouter ici l'interpolation de patchtarget si nécessaire(il faudra ajuster soruce2target du coup)

        float zncc = evaluation_patch(patchsource, patchtarget, imsource, imtarget);
        if (zncc > maxzncc) {
            maxzncc = zncc;
            depthmax = depth;
            patchmax = patchtarget.clone();
        }
        depth += depthstep;
    }
    //debugPatch(patchsource, patchmax, imsource, imtarget,5.5,true);
    if (maxzncc == -std::numeric_limits<double>::infinity()){
        maxzncc = 0;
    }
    debugtab[0] = maxzncc;
    if (depthmax == 0){
        debugtab[1] = -0.1;
    }
    else{
        debugtab[1] = fabs(depthmax-depthinit);
    }
    return depthmax;
}


void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor,bool color) {
    // Vérification que patchsource et patchtarget sont valides (N*2)
    if (patchsource.cols != 2 || patchtarget.cols != 2) {
        std::cout << "patchsource ou patchtarget n'ont pas la bonne taille." << std::endl;
        cout << "patchsource : " << patchsource.size() << endl;
        cout << "patchtarget : " << patchtarget.size() << endl;
        return;
    }

    cv::Mat imsource_resized, imtarget_resized;
    cv::resize(imsource, imsource_resized, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(imtarget, imtarget_resized, cv::Size(), scaleFactor, scaleFactor);

    // Dessiner les points de patchsource sur imsource_resized
    if (color) {
        for (int i = 0; i < patchsource.rows; ++i) {
            cv::circle(imsource_resized, cv::Point(patchsource.at<float>(i, 0) * scaleFactor, patchsource.at<float>(i, 1) * scaleFactor), 5, cv::Scalar(255, 0, 0), -1);
        }
        for (int i = 0; i < patchtarget.rows; ++i) {
            cv::circle(imtarget_resized, cv::Point(patchtarget.at<float>(i, 0) * scaleFactor, patchtarget.at<float>(i, 1) * scaleFactor), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    else{
        for (int i = 0; i < patchsource.rows; ++i) {
            cv::circle(imsource_resized, cv::Point(patchsource.at<float>(i, 0) * scaleFactor, patchsource.at<float>(i, 1) * scaleFactor), 5, cv::Scalar(0, 255, 0), -1);
        }
        for (int i = 0; i < patchtarget.rows; ++i) {
            cv::circle(imtarget_resized, cv::Point(patchtarget.at<float>(i, 0) * scaleFactor, patchtarget.at<float>(i, 1) * scaleFactor), 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    // Fusionner les deux images 
    cv::Mat combined;
    cv::hconcat(imsource_resized, imtarget_resized, combined);

    // Affichage de la fenêtre combinée
    cv::namedWindow("Debug Patch", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Debug Patch", combined); 
    cv::waitKey(0); 
}
