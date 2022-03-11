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
class CTfLiteDepthEstimation : public CTfLiteInterpreter
{
public:
    CTfLiteDepthEstimation();
    CTfLiteDepthEstimation(int width, int height, int ch);
    CTfLiteDepthEstimation(int width, int height, int ch, cv::Scalar mean, cv::Scalar std);

    virtual cv::Mat Run(const std::string& strImgPath, bool isInvertScale);
    void Evaluate(const std::string& strDbPath, const std::string& strSavePath, bool isInvertScale = false);
    cv::Mat DrawDepth(const cv::Mat& depthMat);

private:

    cv::Mat LoadNyuDepthGtFromCsv(std::string& strGtPath);

    float GetAbsRel(const cv::Mat& predict, const cv::Mat& gt);
    float GetRmsError(const cv::Mat& predict, const cv::Mat& gt);
    float GetSiRmsError(const cv::Mat& predict, const cv::Mat& gt);

    cv::Scalar m_mean;
    cv::Scalar m_std;
};
// *******************************************************************************

#endif //TFLITEINTERPRETER_H
