#include <iostream>
#include <fstream>
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

int main() {
    std::cout << "Hello, World!" << std::endl;
    saveEngineToDisk("../yolov5s.onnx", "../yolov5s_engine.trt");
    return 0;
}
