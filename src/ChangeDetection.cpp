#include <memory>
#include <chrono>
#include <iostream>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "ChangeDetection.h"
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"

cv::Mat imgResize2Big(cv::Mat image, int splitCols=256, int splitRows=256)
{
    int imageCols = image.cols;
    int imageRows = image.rows;
    int numCols = imageCols / splitCols;
    int numRows = imageRows / splitRows;
    if (imageCols % splitCols != 0)
    {
        numCols++;
    }
    if (imageRows % splitRows != 0)
    {
        numRows++;
    }
    int image_resizedCols = numCols * splitCols;
    int image_resizedRows = numRows * splitRows;
    cv::Mat image_resized(image_resizedRows, image_resizedCols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat roi = image_resized(cv::Rect(0, 0, imageCols, imageRows));
    image.copyTo(roi);
    return image_resized;
}

cv::Mat imgChange(cv::Mat image1, cv::Mat image2, std::string onnxpath)
{
  cv::Mat frame1 = image1;
  cv::Mat frame2 = image2;
  std::vector<cv::Mat> v1;
  std::vector<cv::Mat> v2;
  int input_width = 256, input_height = 256;
  std::vector<cv::Mat> frame;
  frame.emplace_back(frame1);
  frame.emplace_back(frame2);
  // 创建InferSession, 查询支持硬件设备
  // GPU Mode, 0 - gpu device idrm
  std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
  Ort::SessionOptions session_options;
  Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "exportedonnx");
  // std::cout << __LINE__ << "  " << onnxpath << "  " << (ORTCHAR_T *)modelPath.c_str() << std::endl;

  session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
  Ort::Session session_(env, (ORTCHAR_T *)onnxpath.c_str(), session_options);
  std::vector<std::string> input_node_names;
  std::vector<std::string> output_node_names;

  size_t numInputNodes = session_.GetInputCount();
  size_t numOutputNodes = session_.GetOutputCount();
  // std::cout << __LINE__ << std::endl;
  // std::cout << numInputNodes << "  " << numOutputNodes << std::endl;
  Ort::AllocatorWithDefaultOptions allocator;
  input_node_names.reserve(numInputNodes);

  // 获取输入信息
  int input_w = 0;
  int input_h = 0;
  for (int i = 0; i < numInputNodes; i++)
  {
    auto input_name = session_.GetInputNameAllocated(i, allocator);
    input_node_names.push_back(input_name.get());
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();

    input_w = input_dims[3];
    input_h = input_dims[2];
    // std::cout << "input " << input_node_names[i] << " format: NxCxHxW = ";
    // for (int j = 0; j < input_dims.size(); j++)
    // {
    //   std::cout << input_dims[j] << "x";
    // }
    // std::cout << std::endl;
  }

  // 获取输出信息
  int output_h = 0;
  int output_w = 0;
  for (int i = 0; i < numOutputNodes; i++)
  {
    auto out_name = session_.GetOutputNameAllocated(i, allocator);
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[0]; // 513
    output_w = output_dims[1]; // 513
    output_node_names.push_back(out_name.get());
    // std::cout << "output " << output_node_names[i] << " format: NxCxHxW = ";
    // for (int j = 0; j < output_dims.size(); j++)
    // {
    //   std::cout << output_dims[j] << "x";
    // }
    // std::cout << std::endl;
  }
  std::vector<Ort::Value> input_tensor_vector;
  // format frame
  for (size_t i = 0; i < 2; i++)
  {
    int w = frame[i].cols;
    int h = frame[i].rows;
    float scale_x = input_width / (float)w;
    float scale_y = input_height / (float)h;
    float scale = std::min(scale_x, scale_y);

    int input_batch = 1;
    int input_channel = 3;
c

    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{1, 3, input_h, input_w};
    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_vector.emplace_back(Ort::Value::CreateTensor<float>(allocator_info, input_data_host, tpixels, input_shape_info.data(), input_shape_info.size()));
  }

  const std::array<const char *, 2> inputNames = {input_node_names[0].c_str(), input_node_names[1].c_str()};
  const std::array<const char *, 5> outNames = {output_node_names[0].c_str(), output_node_names[1].c_str(), output_node_names[2].c_str(),
                                                output_node_names[03].c_str(), output_node_names[4].c_str()};
  std::vector<Ort::Value> ort_outputs;
  // std::cout << __LINE__ << std::endl;
  try
  {
    ort_outputs = session_.Run(Ort::RunOptions{nullptr}, inputNames.data(), input_tensor_vector.data(), input_tensor_vector.size(), outNames.data(), outNames.size());
  }
  catch (std::exception &e)
  {
    // std::cout << e.what() << std::endl;
  }
  float *pdata = ort_outputs[4].GetTensorMutableData<float>();
  cv::Mat output(input_width, input_height, CV_8UC1, cv::Scalar(0));
  uchar *d = output.data;
  for (size_t i = 0, j = 0; j < output.rows * output.cols; j++)
  {
    *d++ = pdata[j] > pdata[output.rows * output.cols + j] ? 0 : 255;
  }
  return output;
  session_options.release();
    session_.release();
}

void imgSplit(cv::Mat image1, cv::Mat image2, int splitCols=256, int splitRows=256)
{

    int imageCols = image1.cols;
    int imageRows = image1.rows;
    int numCols = imageCols / splitCols;
    int numRows = imageRows / splitRows;
    for (int y = 0; y < numRows; y++)
    {
        for (int x = 0; x < numCols; x++)
        {
            int startX = x * splitCols;
            int startY = y * splitRows;
            int endX = startX + splitCols;
            int endY = startY + splitRows;
            cv::Mat smallImage1 = image1(cv::Rect(startX, startY, endX - startX, endY - startY));
            cv::imwrite("./tmpA"+std::to_string(x)+std::to_string(y)+".jpg", smallImage1);
            cv::Mat smallImage2 = image2(cv::Rect(startX, startY, endX - startX, endY - startY));
            cv::imwrite("./tmpB"+std::to_string(x)+std::to_string(y)+".jpg", smallImage2);
        }
    }
}

cv::Mat imgResize2Normal(cv::Mat image, int image_oriCols, int image_oriRows)
{
    cv::Mat image_normal = image(cv::Rect(0, 0, image_oriCols, image_oriRows));
    return image_normal;
}

cv::Mat imgResult(cv::Mat image1, cv::Mat image2, std::string onnxpath, int splitCols=256, int splitRows=256)
{
    cv::Mat image1_resized = imgResize2Big(image1);
    cv::Mat image2_resized = imgResize2Big(image2);
    imgSplit(image1_resized, image2_resized);
    int imageCols = image1_resized.cols;
    int imageRows = image1_resized.rows;
    int numCols = imageCols / splitCols;
    int numRows = imageRows / splitRows;
    cv::Mat imageResult(imageRows, imageCols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < numRows; y++)
    {
        for (int x = 0; x < numCols; x++)
        {
            int startX = x * 256;
            int startY = y * 256;
            int endX = startX + 256;
            int endY = startY + 256;
            cv::Mat smallImage1 = cv::imread("./tmpA"+std::to_string(x)+std::to_string(y)+".jpg");
            cv::Mat smallImage2 = cv::imread("./tmpB"+std::to_string(x)+std::to_string(y)+".jpg");
            cv::Mat smallImage_result = imgChange(smallImage1, smallImage2, onnxpath);
            smallImage_result.copyTo(imageResult(cv::Rect(startX, startY, endX - startX, endY - startY)));
        }
    }
    int cols = image1.cols;
    int rows = image1.rows;
    cv::Mat imageResult_resized = imgResize2Normal(imageResult, cols, rows);

    return imageResult_resized;
}

cv::Mat getRegionLabels(cv::Mat image, cv::Mat result)
{
    // 计算变化区域位置
    cv::Mat grayImage;
    cv::cvtColor(result, grayImage, cv::COLOR_BGR2GRAY);
    cv::Mat changeRegion;
    cv::threshold(grayImage, changeRegion, 30, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(changeRegion, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::cout << contours.size() << std::endl;
    // 判断变化区域种类
    for (size_t i = 0; i < contours.size(); ++i)
    {
        int greenCount = 0;
        double greenPercentage = 0;
        cv::Rect bounds = cv::boundingRect(contours[i]);
        cv::Mat roi = image(bounds);
        for (int y = 0; y < roi.rows; ++y)
        {
            for (int x = 0; x < roi.cols; ++x)
            {
                cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
                if (pixel[1] > 50 && pixel[0] < 100 && pixel[2] < 100)
                {
                    greenCount++;
                }
            }
        }
        greenPercentage = static_cast<double>(greenCount) / (roi.rows * roi.cols);
        std::cout << greenPercentage << std::endl;
        std::string label;
        if (greenPercentage > 0.5)
        {
            label = "Grassland";
        }
        else
        {
            label = "Buinding";
        }
        cv::Point textPosition(bounds.tl().x, bounds.tl().y - 10);
        cv::putText(result, label, textPosition, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 1);
    }
    return result;
}


cv::Mat changedetectionTra(cv::Mat image1, cv::Mat image2)
{
    cv::Mat result;
    cv::absdiff(image1, image2, result);
    return result;
}

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class SampleYOLOV7 {
    public:
        SampleYOLOV7(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
        Result InitResource();
        Result ProcessInput(string testImgPath);
        Result Inference(std::vector<InferenceOutput>& inferOutputs);
        Result GetResult(std::vector<InferenceOutput>& inferOutputs, string imagePath, size_t imageIndex, bool release);
        ~SampleYOLOV7();
    private:
        void ReleaseResource();
        AclLiteResource aclResource_;
        AclLiteImageProc imageProcess_;
        AclLiteModel model_;
        aclrtRunMode runMode_;
        ImageData resizedImage_;
        const char *modelPath_;
        int32_t modelWidth_;
        int32_t modelHeight_;
};

SampleYOLOV7::SampleYOLOV7(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) :
                           modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

Result SampleYOLOV7::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV7::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV7::Inference(std::vector<InferenceOutput>& inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
} 

Result SampleYOLOV7::GetResult(std::vector<InferenceOutput>& inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *resBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    

}
void SampleYOLOV7::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}
