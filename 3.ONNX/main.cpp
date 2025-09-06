#include <algorithm>
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
  try {
    // 1️⃣ 创建 ONNX Runtime 环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_test");
    Ort::SessionOptions session_options;

    // 2️⃣ 配置 GPU
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(
        cuda_options); // ✅ 引用，不是指针

    // 3️⃣ 加载模型
    Ort::Session session(env, L"../model.onnx", session_options);
    std::cout << "ONNX Runtime GPU 推理环境就绪 ✅" << std::endl;

    // 4️⃣ 读取图像
    std::string img_path =
        "C:/Users/27427/Desktop/code/AI_infer_learn/MNIST/png/test/0.png";
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
      std::cerr << "❌ 图像未找到: " << img_path << std::endl;
      return -1;
    }
    cv::resize(img, img, cv::Size(28, 28));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // 5️⃣ 创建输入张量
    std::vector<int64_t> input_shape = {1, 1, 28, 28};
    std::vector<float> input_tensor_values(img.total());
    std::memcpy(input_tensor_values.data(), img.data,
                img.total() * sizeof(float));

    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // 6️⃣ 获取输入输出名字
    Ort::AllocatedStringPtr input_name_ptr =
        session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr =
        session.GetOutputNameAllocated(0, allocator);
    const char *input_names[] = {input_name_ptr.get()};
    const char *output_names[] = {output_name_ptr.get()};

    // 7️⃣ 推理
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

    // 8️⃣ 解析输出
    float *output_arr = output_tensors.front().GetTensorMutableData<float>();
    int pred = std::max_element(output_arr, output_arr + 10) - output_arr;
    std::cout << "预测结果: " << pred << std::endl;

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
    return -1;
  }

  return 0;
}
