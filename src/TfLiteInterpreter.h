#ifndef TFLITEINTERPRETER_H
#define TFLITEINTERPRETER_H

#include <fstream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"

class CTfLiteInterpreter
{
private:
    std::unique_ptr <tflite::FlatBufferModel> m_pModel;
    tflite::ops::builtin::BuiltinOpResolver m_resolver;
    std::unique_ptr <tflite::Interpreter> m_pInterpreter;

public:
    CTfLiteInterpreter(int inWidth, int inHeight, int inChannels);
    bool LoadModel(const std::string& strModelPath);
    virtual void Run() {}


    virtual ~CTfLiteInterpreter() {}
protected:
    virtual TfLiteTensor* Interpret(cv::Mat& image);
    virtual cv::Mat ConvertInput(const cv::Mat& srcImg);

    std::vector<int> GetSortedIndex(int numElement, float* pData);

    int m_inWidth;
    int m_inHeight;
    int m_inChannels;

};

// *******************************************************************************

// reference model https://tfhub.dev/intel/midas/v2/2

static constexpr int MIDAS_DEFAULT_IN_COLS = 384; // small 256
static constexpr int MIDAS_DEFAULT_IN_ROWS = 384;
static constexpr int MIDAS_DEFAULT_IN_CH = 3;

static constexpr int NUM_NYUDEPTHV2_TEST_SET = 654;

static cv::Scalar MIDAS_DEFAULT_MEAN = cv::Scalar(0.485, 0.456, 0.406);
static cv::Scalar MIDAS_DEFAULT_STD = cv::Scalar(0.229, 0.224, 0.225);

class CTfLiteDepthEstimation : public CTfLiteInterpreter
{
public:
    CTfLiteDepthEstimation()
            : CTfLiteInterpreter(MIDAS_DEFAULT_IN_COLS, MIDAS_DEFAULT_IN_ROWS, MIDAS_DEFAULT_IN_CH),
              m_mean(MIDAS_DEFAULT_MEAN), m_std(MIDAS_DEFAULT_STD) {}
    CTfLiteDepthEstimation(int width, int height, int ch)
            : CTfLiteInterpreter(width, height, ch),
              m_mean(MIDAS_DEFAULT_MEAN), m_std(MIDAS_DEFAULT_STD) {}
    CTfLiteDepthEstimation(int width, int height, int ch, cv::Scalar mean, cv::Scalar std)
            : CTfLiteInterpreter(width, height, ch),
              m_mean(mean), m_std(std) {}

    virtual cv::Mat Run(const std::string& strImgPath);
    void Evaluate(const std::string& strDbPath, const std::string& strSavePath);
    cv::Mat DrawDepth(const cv::Mat& depthMat);

private:

    cv::Mat LoadNyuDepthGtFromCsv(std::string& strGtPath);

    float GetAbsRel(const cv::Mat& predict, const cv::Mat& gt);
    float GetRmsError(const cv::Mat& predict, const cv::Mat& gt);

    cv::Scalar m_mean;
    cv::Scalar m_std;
};
// *******************************************************************************

#endif //TFLITEINTERPRETER_H

