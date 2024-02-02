#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void afficher_patch(const cv::Mat& patch, const cv::Mat& depths, const cv::Mat& im, const float opacity){
    cv::Mat imaff = im.clone();
    cv::Mat colormap;
    cv::applyColorMap(cv::Mat(depths), colormap, cv::COLORMAP_JET);


    for (int i = 0; i < patch.rows; ++i) {
        cv::Vec3b& color = colormap.at<cv::Vec3b>(i);
        color = color * opacity + imaff.at<cv::Vec3b>(patch.at<int>(i, 0), patch.at<int>(i, 1)) * (1-opacity);
        imaff.at<cv::Vec3b>(patch.at<int>(i, 0), patch.at<int>(i, 1)) = color;
    }

    cv::imshow("Image avec patch 3D et profondeurs", imaff);
    cv::waitKey(0);

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