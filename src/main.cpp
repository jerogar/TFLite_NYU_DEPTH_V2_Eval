#include <iostream>
#include <memory>

#include "TfLiteInterpreter.h"

int main(int argc, char* argv[])
{
    std::string strModelPath = "../res/lite-model_midas_v2_1_small_1_lite_1.tflite";
    std::string strNyuDepthPath = "../res/DenseDepth";
    std::string strSavePath = "../res/result";
    bool isInvertScale = true;

    for (int idx = 0; idx < argc - 1; idx++)
    {
        if (strcmp(argv[idx], "-ModelPath") == 0)
        {
            strModelPath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-dbPath") == 0)
        {
            strNyuDepthPath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-savePath") == 0)
        {
            strSavePath = argv[idx + 1];
        }
        if (strcmp(argv[idx], "-invertScale") == 0)
        {
            int temp = atoi(argv[idx + 1]);
            isInvertScale = (temp > 0) ? true : false;
        }
    }

    std::cout << "Model: " << strModelPath << "\n";

    // Run Model
    std::unique_ptr <CTfLiteDepthEstimation> pDepthEstimator = std::make_unique<CTfLiteDepthEstimation>();
    pDepthEstimator->LoadModel(strModelPath);
    pDepthEstimator->Evaluate(strNyuDepthPath, strSavePath, isInvertScale);

    return 0;
}

