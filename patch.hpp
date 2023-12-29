#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>


/**
 * crée un patch à partir d'un point et d'une image
 * @param[in] im Image
 * @param[in] point Point
 * @param[out] patch Patch
 * @param[in] size Taille du patch
*/
void create_patch(const cv::Mat& im, const cv::Vec2f point, cv::Mat& patch, int size);

/**
* Projette dans le repère monde le patch de l'image de reference puis le reprojette dans l'image source
* @param[in] depth Profondeur qu'on teste
* @param[in] K Matrice intrinseque de la camera
* @param[in] P changement de pose entre les deux images, P = (R | t) 4*3 
* @param[in] patchsource Patch de l'image de départ
* @param[out] patchtarget Patch de l'image d'arivee
*/
void source2target(float depth,const cv::Mat& K, const cv::Mat& P, const cv::Mat& patchsource, cv::Mat& patchtarget);


/**
* Calcule la similarité ZNCC entre deux patchs
* @param[in] patchsource Patch de l'image de départ
* @param[in] patchtarget Patch de l'image d'arivee
* @param[in] imsource Image de départ
* @param[in] imtarget Image d'arrivee
* @return Similarité ZNCC
*/ 
float evaluation_patch(const cv::Mat& patchsource, const cv::Mat& patchtarget, const cv::Mat& imsource, const cv::Mat& imtarget);

/**
 * Calcule la profondeur d'un point à partir de deux images et du changement de pose
 * @param[in] point Point dont on veut la profondeur
 * @param[in] depthinit Profondeur initiale
 * @param[in] imsource Image source
 * @param[in] imtarget Image d'arrivée
 * @param[in] K Matrice intrinseque de la camera
 * @param[in] P changement de pose entre les deux images, P = (R | t) 4*3 
 * @return Profondeur du point
*/
float patch_integration(cv::Point2f point, const cv::Mat& imsource, const cv::Mat& normalsource,const cv::Mat& imtarget,float depthinit, const::cv::Mat& K, const cv::Mat& P);