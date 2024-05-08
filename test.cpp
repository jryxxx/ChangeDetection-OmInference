#include <memory>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"

using namespace std;
using namespace cv;

typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

class ChangeDetection
{
public:
    ChangeDetection(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(string testImgPath0, string testImgPath1);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, string imagePath, bool release);
    ~ChangeDetection();

private:
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage0_;
    ImageData resizedImage1_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

ChangeDetection::ChangeDetection(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) : modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

ChangeDetection::~ChangeDetection()
{
    ReleaseResource();
}

Result ChangeDetection::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result ChangeDetection::ProcessInput(string testImgPath0, string testImgPath1)
{
    // read image from file
    AclLiteError ret;
    ImageData image0;
    ret = ReadJpeg(image0, testImgPath0);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }
    ImageData image1;
    ret = ReadJpeg(image1, testImgPath1);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }
    // copy image from host to dvpp
    ImageData imageDevice0;
    ret = CopyImageToDevice(imageDevice0, image0, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }
    ImageData imageDevice1;
    ret = CopyImageToDevice(imageDevice1, image1, runMode_, MEMORY_DVPP);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage0;
    ret = imageProcess_.JpegD(yuvImage0, imageDevice0);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }
    ImageData yuvImage1;
    ret = imageProcess_.JpegD(yuvImage1, imageDevice1);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage0_, yuvImage0, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = imageProcess_.Resize(resizedImage1_, yuvImage1, modelWidth_, modelHeight_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result ChangeDetection::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // 内存大小 -> 256 x 256 x 1.5 = 98304
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage0_.data.get()), resizedImage0_.size,
                                          static_cast<void *>(resizedImage1_.data.get()), resizedImage1_.size);

    cout << "debug -> input's size " << resizedImage0_.size << endl;
    cout << "debug -> input's size " << resizedImage1_.size << endl;
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }
    else
    {
        ACLLITE_LOG_INFO("CreateInput success");
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result ChangeDetection::GetResult(std::vector<InferenceOutput> &inferOutputs,
                                  string imagePath, bool release)
{
    release = false;
    int input_width = 256;
    int input_height = 256;
    uint32_t outputDataBufId = 4;
    float *resBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    std::cout << "Debug -> Output's size is " << inferOutputs[outputDataBufId].size << endl;
    cv::Mat output(input_width, input_height, CV_8UC1, cv::Scalar(0));
    uchar *d = output.data;
    for (size_t j = 0; j < output.rows * output.cols; j++)
    {
        int debug_data = resBuff[j] > resBuff[output.rows * output.cols + j] ? 0 : 255;
        *d++ = resBuff[j] > resBuff[output.rows * output.cols + j] ? 0 : 255;
        // cout << "Debug -> Output data " << debug_data << endl;
    }
    imwrite("tmp_res.png", output);
    imshow("res", output);
    waitKey(0);
    return SUCCESS;
}

void ChangeDetection::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    const char *modelPath = "../weights/DSIFN.om";
    const int32_t modelWidth = 256;
    const int32_t modelHeight = 256;

    string fileName1 = "/home/thtf/test/ChangeDetection/image/test-1.jpeg";
    string fileName2 = "/home/thtf/test/ChangeDetection/image/test-2.jpeg";
    bool release = false;
    std::vector<InferenceOutput> inferOutputs;
    ChangeDetection sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = sampleYOLO.ProcessInput(fileName1, fileName2);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = sampleYOLO.Inference(inferOutputs);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = sampleYOLO.GetResult(inferOutputs, fileName1, release);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
        return FAILED;
    }
    return 0;
}
