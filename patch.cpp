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
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget) {
    //si les patchs ont la même taille
    //cout << "avant if de evaluation_patch \n";
    //cout << patchsource.size() << " \n";
    //cout << patchtarget.size() << " \n";
    if (patchsource.size() != patchtarget.size()) {
        throw std::invalid_argument("Les patchs doivent être de la même taille");
    }
    Mat srcval,targetval; //a refaire : pushval c nul
    //cout << "avant for de evaluation_patch \n";
    //for (int i = 0; i < patchsource.rows; ++i) {
    //    srcval.push_back(imsource.at<float>(static_cast<int>(patchsource.at<float>(i,0)), static_cast<int>(patchsource.at<float>(i,1))));
    //    targetval.push_back(imtarget.at<float>(static_cast<int>(patchtarget.at<float>(i,0)), static_cast<int>(patchtarget.at<float>(i,1))));
    //}

    //cout << "avant meanstddev de evaluation_patch \n";
    cv::Scalar meanSource, stddevSource;
    cv::meanStdDev(patchsource, meanSource, stddevSource);
    cv::Mat normPatchSource = (patchsource - meanSource) / stddevSource;

    cv::Scalar meanTarget, stddevTarget;
    cv::meanStdDev(patchtarget, meanTarget, stddevTarget);
    cv::Mat normPatchTarget = (patchtarget - meanTarget) / stddevTarget;

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
    
    
    //cout << "avant create patch \n";
    create_patch(imsource, point, patchsource, size);
    //cout << "apres create patch \n";
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
    //cout << "avant depthcentre \n";
    //cout << "taille de integratedpatch : "<<integratedpatch.size() << "\n";
    //cout << "size : "<<size*(size/2) - size/2 << "\n";
    float depthcentre = (integratedpatch.at<float>(size*size/2 - size/2,2)); 
    //int deppp = depthcentre.at<int>(2);
    //cout << "depthcentre : "<<depthcentre << "\n";
    //cout << "apres depthcentre"<<" \n";
    //cout << integratedpatch << " 1 \n";
    integratedpatch.col(2) = integratedpatch.col(2)/depthcentre;
    //cout << integratedpatch.col(2) << " 2 \n";

    //a faire : prunage des points qui ne sont pas dans l'image target (comparer normal du point et direction camera) (ou a integrer dans source2target)

    int nb_profondeur = 10; // Nombre de profondeurs testées
    float depthstep = 0.01; // Ou 0.01 à tester
    //cout << "avant findOptimalDepth \n";
    float optimalDepth = findOptimalDepth(depthinit, K, P1,P2, integratedpatch, imsource, imtarget, patchsource, depthstep, nb_profondeur);


    return optimalDepth;
}

/*
* P = (R | t)
*/
void source2target(float depth, const cv::Mat& K, const cv::Mat& P1,const cv::Mat& P2, const cv::Mat& integratedpatch, cv::Mat& patchtarget) {
    // a faire : vectoriser pour eviter les boucles
    for (int i = 0; i < integratedpatch.rows; ++i) {
            //pixel en coordonnées 3D
            patchtarget = Mat::zeros(integratedpatch.rows, 2, CV_32F);
            //cout << K.type() << "TYPE DE K \n";
            //cout << "avant pointR1 \n";
            //cout << (integratedpatch.row(i).t()).size() << "\n"; 
            //cout << K.inv().size() << "\n";
            //cout << (K.inv()) * (integratedpatch.row(i).t()) << "\n";
            
            cv::Mat p1 = cv::Mat(cv::Vec3f(integratedpatch.at<float>(i,0), integratedpatch.at<float>(i,1), 1.0f));
            cv::Mat w1 = (K.inv() * p1);
            cout << "p1 : "<<p1<<"\n";
            cout << "w1 : "<<w1<<"\n";
            cv::Mat pointR1 = depth*integratedpatch.at<float>(i,2) * w1;
            cv::Mat R1 = P1.rowRange(0, 3).colRange(0, 3);
            cv::Mat t1 = P1.rowRange(0, 3).col(3);
            cout << "R1 : "<<R1<<"\n";
            cout << "t1 : "<<t1<<"\n";
            cout << "P2 : "<<P2<<"\n";
            cout << "K : "<<K<<"\n";
            cv:Mat pointWorld = R1.inv() * pointR1 + t1; // ICI MULTIPLIER PAR R1^-1 ET EN DESSOUS PAR R2 (il faudra donc charger deux matrices de rot, car la rot entre 1 et 2 sera R1^-1 * R2 (de meme pour t car on les centres))
            cv::Mat pointR2 = P2 * cv::Mat(cv::Vec4f(pointR1.at<float>(0), pointR1.at<float>(1), pointR1.at<float>(2), 1.0f)); //changer les trucs vec vec4f etc..
            
            cout << "pointR1 : "<<pointR1<<"\n";
            cout << "pointR2 : "<<pointR2<<"\n";
            cv::Mat w2 = pointR2 / pointR2.at<float>(2);
            cv::Mat pp2 = K * w2;
            cout << "w2 : "<<w2<<"\n";
            cout << "pp2 : "<<pp2<<"\n";
            cv::Mat p2 = pp2.rowRange(0, 2);
            //cout << "p2 size : "<<p2.size()<<"\n";

            //coordonnées pixel entieres(peut etre pas necessaire/pas bien)         
            //cout << "avant targetX \n";
            int targetX = static_cast<int>(p2.at<float>(0));
            int targetY = static_cast<int>(p2.at<float>(1));
            cout << "targetX : "<<targetX<<"\n";
            cout << "targetY : "<<targetY<<"\n";

            //cout << "P : "<<P1<<"\n";
            //cout << "pattarget cols : "<<patchtarget.cols<<"\n";
            //cout << "pattarget rows : "<<patchtarget.rows<<"\n";
            //le point projeté est à l'intérieur des limites de l'image
            //cout << "avant if de source to target \n";
            //if (targetX >= 0 && targetX < patchtarget.cols && targetY >= 0 && targetY < patchtarget.rows) {
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
    //cout << "x1 : "<<x1<<"\n";
    //cout << "y1 : "<<y1<<"\n";
    //cout << "x2 : "<<x2<<"\n";
    //cout << "y2 : "<<y2<<"\n";
    /*Creation finale du patch qui sera donc les POSITIONS dans le repère IMAGE des pixels concerné*/
    int idx =0;
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
    float depth = depthinit - (nb_profondeur / 2) * depthstep;
    cv::Mat patchmax;
    for(int i = 1; i < nb_profondeur; i++) {
        depth += i * depthstep;
        cv::Mat patchtarget;
        
        source2target(depth, K, P1,P2, integratedpatch, patchtarget);
        //cout << "apres source2target \n";
        // Ajouter ici l'interpolation de patchtarget si nécessaire

        float zncc = evaluation_patch(patchsource, patchtarget, imsource, imtarget);
        if (zncc > maxzncc) {
            maxzncc = zncc;
            depthmax = depth;
            patchmax = patchtarget.clone();
            //cout << patchtarget << " \n";
        }
        //cout <<"apres if de source to target \n";
    }

    int x_source, y_source, x_target, y_target;
    x_source = y_source = x_target = y_target = 10; // exemple de positions, à ajuster selon vos besoins
    
    debugPatch(patchsource, patchmax, imsource, imtarget,10.5);

    return depthmax;
}


void debugPatch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget, double scaleFactor) {
    // Vérification que patchsource et patchtarget sont valides (N*2)
    if (patchsource.cols != 2 || patchtarget.cols != 2) {
        std::cout << "patchsource ou patchtarget n'ont pas la bonne taille." << std::endl;
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
        cv::circle(imsource_resized, cv::Point(patchsource.at<float>(i, 0) * scaleFactor, patchsource.at<float>(i, 1) * scaleFactor), 2, cv::Scalar(0, 255, 0), -1);
    }

    // Dessiner les points de patchtarget sur imtarget_resized
    for (int i = 0; i < patchtarget.rows; ++i) {
        cv::circle(imtarget_resized, cv::Point(patchtarget.at<float>(i, 0) * scaleFactor, patchtarget.at<float>(i, 1) * scaleFactor), 2, cv::Scalar(0, 255, 0), -1);
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
    cv::waitKey(0); // Attendre une touche pour fermer la fenêtre
}
