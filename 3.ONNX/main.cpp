#include "onnxruntime_c_api.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
  // 1️⃣ 创建 ONNX Runtime 环境
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx test");
  Ort::SessionOptions sessop;
  // 2️⃣ 配置执行提供者（使用 CPU，如需 GPU 请取消注释下一行）
  // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessop,
  // 0));
  sessop.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // 3️⃣ 加载模型
  Ort::Session session(env, L"cnn_onnx_module.onnx", sessop);

  // 4️⃣ 读取图像
  std::string imgPath = "test_image.png"; // 需要提供测试图像
  cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
  if (img.empty()) {
    std::cerr << "错误: 无法读取图像文件 " << imgPath << std::endl;
    std::cerr << "请提供一个 28x28 的灰度图像用于测试" << std::endl;
    return -1;
  }
  cv::resize(img, img, cv::Size(28, 28));
  img.convertTo(img, CV_32F, 1.0 / 255);

  // 5️⃣ 创建输入张量
  std::vector<int64_t> input_shape{1, 1, 28, 28};
  std::vector<float> input_val(img.total());
  std::memcpy(input_val.data(), img.data, img.total() * sizeof(float));

  Ort::AllocatorWithDefaultOptions alloctor;
  auto memort_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memort_info, input_val.data(), input_val.size(), input_shape.data(),
      input_shape.size());

  // 6️⃣ 获取输入输出名字
  Ort::AllocatedStringPtr input_name_ptr =
      session.GetInputNameAllocated(0, alloctor);
  Ort::AllocatedStringPtr output_name_ptr =
      session.GetOutputNameAllocated(0, alloctor);
  const char *input_names[] = {input_name_ptr.get()};
  const char *output_names[] = {output_name_ptr.get()};

  // 7️⃣ 推理
  auto output = session.Run(Ort::RunOptions{nullptr}, input_names,
                            &input_tensor, 1, output_names, 1);

  // 8️⃣ 解析输出

  float *output_arr = output.front().GetTensorMutableData<float>();
  int pred = std::max_element(output_arr, output_arr + 10) - output_arr;
  std::cout << "预测结果: " << pred << std::endl;
}
