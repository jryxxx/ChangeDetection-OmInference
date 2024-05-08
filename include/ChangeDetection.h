#ifndef CHANGEDETECYION_H
#define CHANGEDETECYION_H

#include <memory>
#include <chrono>
#include <string.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"

cv::Mat imgResize2Big(cv::Mat image, int splitCols, int splitRows);
cv::Mat imgChange(cv::Mat image1, cv::Mat image2, std::string onnxpath);
void imgSplit(cv::Mat image1, cv::Mat image2, int splitCols, int splitRows);
cv::Mat imgResize2Normal(cv::Mat image, int image_oriCols, int image_oriRows);
cv::Mat imgResult(cv::Mat image1, cv::Mat image2, std::string onnxpath, int splitCols, int splitRows);
cv::Mat changedetectionTra(cv::Mat image1, cv::Mat image2);
cv::Mat getRegionLabels(cv::Mat image, cv::Mat result);

#endif // CHANGEDETECYION_H
