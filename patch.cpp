#include "patch.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include "normals.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

float findOptimalDepth(float depthinit, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, const cv::Mat& imsource, const cv::Mat& imtarget, const cv::Mat& patchsource, float depthstep, int nb_profondeur);
void vectorize_patch(const cv::Mat& integratedPatch, cv::Mat& vectorizedPatch);
void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor=1.5);

//Une amélioration pourrait être d'utiliser autre chose que zncc pour déterminer la similarité entre deux patchs
//ZNCC A REFAIRE CAR LA CA PREND LES COORDONNE ET PAS LES VALEURS
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget) {
    if (patchsource.size() != patchtarget.size() || patchsource.rows == 0 || patchsource.cols != 2) {
        throw std::invalid_argument("Les patchs doivent être de la même taille et non vides avec 2 colonnes");
    }

    // Vérifiez si les images sources et cibles sont de type CV_8UC3
    if (imsource.type() != CV_8UC3 || imtarget.type() != CV_8UC3) {
        throw std::invalid_argument("Les images sources et cibles doivent être de type CV_8UC3");
    }

    std::vector<float> srcVals, targetVals;
    srcVals.reserve(patchsource.rows);
    targetVals.reserve(patchtarget.rows);

    // Récupérer les valeurs correspondantes en niveau de gris dans les images.
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

    cv::Mat srcMat = cv::Mat(srcVals).reshape(1, 1);
    cv::Mat targetMat = cv::Mat(targetVals).reshape(1, 1);

    cv::Scalar meanSource, stddevSource;
    cv::meanStdDev(srcMat, meanSource, stddevSource);
    cv::Mat normPatchSource = (srcMat - meanSource) / stddevSource;

    cv::Scalar meanTarget, stddevTarget;
    cv::meanStdDev(targetMat, meanTarget, stddevTarget);
    cv::Mat normPatchTarget = (targetMat - meanTarget) / stddevTarget;

    cv::Mat result;
    cv::multiply(normPatchSource, normPatchTarget, result);
    
    float zncc = cv::sum(result)[0];

    return zncc;
}




//a refacto (param et init)
float patch_integration(Point2f point, const cv::Mat& imsource, const cv::Mat& normalsource,const cv::Mat& imtarget,float depthinit, const::cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2,bool debug, const cv::Mat& depthmap){
    Mat patchsource, patchtarget, patchnormalsource,prepatchnormalsource,integratedpatch;
    cv::Mat preintegratedpatch = Mat::zeros(patchsource.rows, 3, CV_32F);
    int size = 5;
    
    

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

    int nb_profondeur = 5; // Nombre de profondeurs testées
    float depthstep = 0.01; // Ou 0.01 à tester
    float optimalDepth = findOptimalDepth(depthinit, K, P1,P2, integratedpatch, imsource, imtarget, patchsource, depthstep, nb_profondeur);


    return optimalDepth;
}

/*
* P = (R | t)
*/
void source2target(float depth, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, cv::Mat& patchtarget,int boundcols, int boundrows) {
    // a faire : vectoriser pour eviter les boucles
    patchtarget = Mat::zeros(integratedpatch.rows, 2, CV_32F);
    for (int i = 0; i < integratedpatch.rows; ++i) {
            //pixel en coordonnées 3D
            cv::Mat p1 = cv::Mat(cv::Vec3f(integratedpatch.at<float>(i,0), integratedpatch.at<float>(i,1), 1.0f));
            cv::Mat w1 = (K.inv() * p1);
            cv::Mat pointR1 = depth * integratedpatch.at<float>(i,2) * w1;
            cv::Mat R2 = P2.rowRange(0, 3).colRange(0, 3);
            cv::Mat t2 = P2.rowRange(0, 3).col(3);
            cv:Mat pointWorld = P1 * cv::Mat(cv::Vec4f(pointR1.at<float>(0), pointR1.at<float>(1), pointR1.at<float>(2), 1.0f));
            cv::Mat pointR2 = R2.t() * pointWorld - R2.t() * t2;
            cv::Mat w2 = pointR2 ;
            cv::Mat pp2 = K * w2;
            pp2 = pp2 / pp2.at<float>(2);

            cv::Mat p2 = pp2.rowRange(0, 2);
            //coordonnées pixel entieres(peut etre pas necessaire/pas bien)         
            
            int targetX = static_cast<int>(p2.at<float>(0));
            int targetY = static_cast<int>(p2.at<float>(1));

            if (targetX < 0 || targetX >= boundcols || targetY < 0 || targetY >= boundrows) {
                continue;
            }


            patchtarget.at<float>(i,0) = targetX; // 
            patchtarget.at<float>(i,1) = targetY;
       
            //}
        
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
        x2 = size-1;
    }
    if (y1 < 0){
        y1 = 0;
        y2 = size-1;
    }
    if (x2 >= im.cols){
        x2 = im.cols-1;
        x1 = im.cols - size ;
    }
    if (y2 >= im.rows){
        y2 = im.rows-1;
        y1 = im.rows - size;
    }
    /*Creation finale du patch qui sera donc les POSITIONS dans le repère IMAGE des pixels concerné*/
    int idx =0;
    float scale = 5.0;
    if (keepval){
        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat im_with_rect = im.clone();
        cv::rectangle(im_with_rect, roi, cv::Scalar(0, 255, 0), 2); // Green rectangle with thickness 2

        // Rescale the image with rectangle if scale is not 1.0
        if (scale != 1.0) {
            cv::resize(im_with_rect, im_with_rect, cv::Size(), scale, scale);
        }

        // Display the original image with the patch rectangle
        cv::namedWindow("Image with Patch", cv::WINDOW_AUTOSIZE);
        cv::imshow("Image with Patch", im_with_rect);
        cv::waitKey(0); // Wait for a key press
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

void vectorize_patch(const cv::Mat& integratedPatch, cv::Mat& vectorizedPatch) {
    int size = integratedPatch.rows; // Assumons que integratedPatch est une matrice carrée
    vectorizedPatch = Mat::zeros(size*size, 3, CV_32FC1); // Matrice pour stocker x, y, et la valeur du pixel

    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            float pixelValue = integratedPatch.at<float>(x, y);
            vectorizedPatch.at<Vec3f>(x*size + y) = Vec3f(static_cast<float>(x), static_cast<float>(y), pixelValue);
        }
    }
}

float findOptimalDepth(float depthinit, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, const cv::Mat& imsource, const cv::Mat& imtarget, const cv::Mat& patchsource, float depthstep, int nb_profondeur) {
    float depthmax = 0;
    float maxzncc = 0;
    float depth = (depthinit - (nb_profondeur / 2) * depthstep);
    cv::Mat patchmax;
    for(int i = 0; i < nb_profondeur; i++) {
        cv::Mat patchtarget;
        source2target(depth, K, P1,P2, integratedpatch, patchtarget,imtarget.cols, imtarget.rows);
        if (patchtarget.rows == 0) {
            continue;
        }
        // Ajouter ici l'interpolation de patchtarget si nécessaire

        float zncc = evaluation_patch(patchsource, patchtarget, imsource, imtarget);
        cout << "zncc : " << zncc <<  " depth : " << depth <<std::endl;
        if (zncc > maxzncc) {
            maxzncc = zncc;
            depthmax = depth;
            patchmax = patchtarget.clone();
        }

        depth += depthstep;
    }

    int x_source, y_source, x_target, y_target;
    x_source = y_source = x_target = y_target = 10; // exemple de positions, à ajuster selon vos besoins
    cout << "depthinit : "<<depthinit<< " depthamx : " << depthmax << "\n";
    debugPatch(patchsource, patchmax, imsource, imtarget,5.5);

    return depthmax;
}


void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor) {
    // Vérification que patchsource et patchtarget sont valides (N*2)
    if (patchsource.cols != 2 || patchtarget.cols != 2) {
        std::cout << "patchsource ou patchtarget n'ont pas la bonne taille." << std::endl;
        cout << "patchsource : " << patchsource.size() << endl;
        cout << "patchtarget : " << patchtarget.size() << endl;
        return;
    }

    // Calcul des rectangles englobants pour patchsource et patchtarget
    cv::Rect roiSource = cv::boundingRect(patchsource);
    cv::Rect roiTarget = cv::boundingRect(patchtarget);

    // Redimensionnement des images pour agrandir
    cv::Mat imsource_resized, imtarget_resized;
    cv::resize(imsource, imsource_resized, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(imtarget, imtarget_resized, cv::Size(), scaleFactor, scaleFactor);

    // Dessiner les points de patchsource sur imsource_resized
    for (int i = 0; i < patchsource.rows; ++i) {
        cv::circle(imsource_resized, cv::Point(patchsource.at<float>(i, 0) * scaleFactor, patchsource.at<float>(i, 1) * scaleFactor), 5, cv::Scalar(0, 255, 0), -1);
    }

    // Dessiner les points de patchtarget sur imtarget_resized
    //cout << "patchtarget : " << patchtarget << endl;
    for (int i = 0; i < patchtarget.rows; ++i) {
        cv::circle(imtarget_resized, cv::Point(patchtarget.at<float>(i, 0) * scaleFactor, patchtarget.at<float>(i, 1) * scaleFactor), 5, cv::Scalar(0, 255, 0), -1);
    }

    // Dessiner un rectangle autour des ROI pour les visualiser plus facilement
    //cv::rectangle(imsource_resized, roiSource, cv::Scalar(0, 255, 0), 2);
    //cv::rectangle(imtarget_resized, roiTarget, cv::Scalar(0, 255, 0), 2);

    // Fusionner les deux images côte à côte dans la même fenêtre
    cv::Mat combined;
    cv::hconcat(imsource_resized, imtarget_resized, combined);

    // Affichage de la fenêtre combinée
    cv::namedWindow("Debug Patch", cv::WINDOW_AUTOSIZE); // Création de la fenêtre
    cv::imshow("Debug Patch", combined); // Affichage de l'image
    //cv::waitKey(0); // Attendre une touche pour fermer la fenêtre
}
