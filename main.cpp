#include <iostream>
#include <fstream>
#include <cassert>
#include <opencv4/opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"

class Logger : public nvinfer1::ILogger
{
    void log (nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        if (severity != nvinfer1::ILogger::Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

void saveEngineToDisk(const std::string& onnxFileName, const std::string& engineName)
{
    std::cout << "[INFO] " << "Creating builder and network..." << std::endl;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    std::cout << "[INFO] " << "Loading ONNX file in the disk..." << std::endl;
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    std::cout << "[INFO] " << "Building cuda engine..." << std::endl;
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 30); // 60 Mb
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    std::cout << "[INFO] " << "Serializing engine..." << std::endl;
    nvinfer1::IHostMemory* serializedModel = engine->serialize();

    std::cout << "[INFO] " << "Saving serialized model in the disk..." << std::endl;
    std::ofstream ofs(engineName, std::ios::binary | std::ios::out);
    ofs.write((char*)(serializedModel->data()), serializedModel->size());
    ofs.close();

    serializedModel->destroy();
}

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

nvinfer1::ICudaEngine* loadEngineFromDisk(const std::string& engineName)
{
    std::cout << "Loading " << engineName << " from the disk." << std::endl;
    std::ifstream ifs(engineName, std::ios::binary);

    if (!ifs.good())
    {
        std::cout << "Please check the path of your engine file." << std::endl;
        exit(-1);
    }

    size_t size = 0;

    ifs.seekg(0, ifs.end);
    size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char* trtModelStream = new char[size];

    ifs.read(trtModelStream, size);
    ifs.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);

    delete[] trtModelStream;
    return engine;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
//    saveEngineToDisk("../yolov5s.onnx", "../yolov5s_engine.trt");
    nvinfer1::ICudaEngine* engine = loadEngineFromDisk("../yolov5s_engine.trt");
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    const int inputW = 640;
    const int inputH = 640;

    cv::Mat image = cv::imread("../zidane.jpg");
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    void* buffers[2] = {NULL, NULL};
    const int OUTPUT_SIZE = 1000 * 6 * sizeof(float) / sizeof(float) + 1;
    const int nBatchSize = 1;
    const int nOutputSize = OUTPUT_SIZE;
    cudaMalloc(&buffers[0], nBatchSize * inputW * inputH * sizeof(float));cudaMalloc(&buffers[1], nBatchSize * nOutputSize * sizeof(float));cudaStream_t stream;
    cudaStreamCreate(&stream);

    static float data[nBatchSize * 3 * inputW * inputH]; // try to flatten your image
//    memcpy(data, image.ptr<float>(0), inputH * inputW * sizeof(float));

    int i = 0;

    cv::Mat preImage = preprocess_img(image, inputW, inputH);
    for (int row = 0; row < inputH; ++row)
    {
        uchar* ucPixel = preImage.data + row * preImage.step; // flatten image
        for (int col = 0; col < inputW; ++col)
        {
            // normalize image and convert BGR to RGB
            data[i] = (float)ucPixel[2] / 255.0;
            data[i + 1 * inputH * inputW] = (float)ucPixel[1] / 255.0;
            data[i + 2 * inputH * inputW] = (float)ucPixel[0] / 255,0;
            ucPixel += 3;
            ++i;
        }
    }

    cudaMemcpyAsync(buffers[0], data, nBatchSize * inputW * inputH * sizeof(float), cudaMemcpyHostToDevice, stream);

    auto start = std::chrono::high_resolution_clock::now();
    context->enqueueV2(buffers, stream, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    float totalTime = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << totalTime << " ms" << std::endl;
    float prob[nBatchSize * nOutputSize];

    cudaMemcpyAsync(prob, buffers[1], 1 * nOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return 0;
}
