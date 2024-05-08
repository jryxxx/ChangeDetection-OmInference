#include <memory>
#include <chrono>

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ChangeDetection.h"

int main()
{
  // 使用深度学习模型
  //  cv::Mat image_change1 = cv::imread("../image/test1.png");
  //  cv::Mat image_change2 = cv::imread("../image/test2.png");
  //  std::string onnxpath = "../weights/DSIFN.onnx";
  //  cv::Mat result = imgResult(image_change1, image_change2, onnxpath, 256, 256);
  //  cv::imwrite("result.jpg", result);
  // 使用传统方法
  // 变化前
  cv::Mat image_change1 = cv::imread("../image/test1.png");
  // 变化后
  cv::Mat image_change2 = cv::imread("../image/test2.png");
  // 变化检测结果
  cv::Mat result = changedetectionTra(image_change1, image_change2);
  // 获取变化区域类别
  cv::Mat labelImage = getRegionLabels(image_change1, result);
  cv::imwrite("../image/result.jpg", result);
  return 0;
}
