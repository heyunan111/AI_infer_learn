#include "activation_layer.h"
#include <cmath>
#include <iostream>
#include <vector>

ActivationLayer::ActivationLayer(enActiveFuncType type, int inputDim,
                                 int outputDim)
    : _type(type), _input_dimension(inputDim), _output_dimension(outputDim) {}

Eigen::VectorXd ActivationLayer::compute(const Eigen::VectorXd &x) {
  if (_type == enActiveFuncType::enSoftMax) {
    return soft_max(x);
  } else if (_type == enActiveFuncType::enReLU) {
    return relu(x);
  }
  return {};
}

Eigen::VectorXd ActivationLayer::soft_max(const Eigen::VectorXd &x) {
  Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
  return exp_x / exp_x.sum();
}
Eigen::VectorXd ActivationLayer::relu(const Eigen::VectorXd &x) {
  return x.array().max(0.0);
}

bool vectors_almost_equal(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                          double tolerance = 1e-6) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tolerance)
      return false;
  }
  return true;
}

void print_test_result(const std::string &test_name, bool passed) {
  std::cout << test_name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

void print_vector(const std::string &name, const Eigen::VectorXd &vec) {
  std::cout << name << ": [";
  for (int i = 0; i < vec.size(); ++i) {
    std::cout << vec[i];
    if (i < vec.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;
}

// soft_max 测试用例
void test_soft_max() {
  std::cout << "=== Testing soft_max ===" << std::endl;

  // 测试用例1: 正数输入
  Eigen::VectorXd input1(3);
  input1 << 1.0, 2.0, 3.0;
  Eigen::VectorXd result1 = ActivationLayer::soft_max(input1);

  // 预期结果: e^1/(e^1+e^2+e^3), e^2/(...), e^3/(...)
  double sum_exp = std::exp(1.0) + std::exp(2.0) + std::exp(3.0);
  Eigen::VectorXd expected1(3);
  expected1 << std::exp(1.0) / sum_exp, std::exp(2.0) / sum_exp,
      std::exp(3.0) / sum_exp;

  print_test_result("Positive numbers",
                    vectors_almost_equal(result1, expected1));
  print_vector("Input", input1);
  print_vector("Result", result1);
  print_vector("Expected", expected1);
  std::cout << "Sum of result: " << result1.sum() << " (should be ~1.0)"
            << std::endl;

  // 测试用例2: 包含负数
  Eigen::VectorXd input2(2);
  input2 << -1.0, 1.0;
  Eigen::VectorXd result2 = ActivationLayer::soft_max(input2);

  double sum_exp2 = std::exp(-1.0) + std::exp(1.0);
  Eigen::VectorXd expected2(2);
  expected2 << std::exp(-1.0) / sum_exp2, std::exp(1.0) / sum_exp2;

  print_test_result("Mixed positive/negative",
                    vectors_almost_equal(result2, expected2));
  print_vector("Input", input2);
  print_vector("Result", result2);
  print_vector("Expected", expected2);
  std::cout << "Sum of result: " << result2.sum() << " (should be ~1.0)"
            << std::endl;

  // 测试用例3: 全负数
  Eigen::VectorXd input3(3);
  input3 << -5.0, -3.0, -1.0;
  Eigen::VectorXd result3 = ActivationLayer::soft_max(input3);

  // 检查sum是否为1
  bool sum_correct = std::abs(result3.sum() - 1.0) < 1e-6;
  print_test_result("All negative numbers sum to 1", sum_correct);
  std::cout << "Sum of result: " << result3.sum() << std::endl;

  // 测试用例4: 大数值（测试数值稳定性）
  Eigen::VectorXd input4(3);
  input4 << 1000.0, 1001.0, 1002.0;
  Eigen::VectorXd result4 = ActivationLayer::soft_max(input4);

  // 应该不会出现NaN或inf
  bool is_valid = !result4.hasNaN() && (result4.array() > 0).all();
  print_test_result("Large numbers (numerical stability)", is_valid);
  print_vector("Large input result", result4);
  std::cout << "Sum of result: " << result4.sum() << std::endl;
}

// relu 测试用例
void test_relu() {
  std::cout << "\n=== Testing relu ===" << std::endl;

  // 测试用例1: 混合正负数
  Eigen::VectorXd input1(4);
  input1 << -2.0, -1.0, 0.0, 3.0;
  Eigen::VectorXd result1 = ActivationLayer::relu(input1);
  Eigen::VectorXd expected1(4);
  expected1 << 0.0, 0.0, 0.0, 3.0;

  print_test_result("Mixed positive/negative/zero",
                    vectors_almost_equal(result1, expected1));
  print_vector("Input", input1);
  print_vector("Result", result1);
  print_vector("Expected", expected1);

  // 测试用例2: 全正数
  Eigen::VectorXd input2(3);
  input2 << 1.0, 2.0, 3.0;
  Eigen::VectorXd result2 = ActivationLayer::relu(input2);

  print_test_result("All positive (identity)",
                    vectors_almost_equal(result2, input2));
  print_vector("Input", input2);
  print_vector("Result", result2);

  // 测试用例3: 全负数
  Eigen::VectorXd input3(3);
  input3 << -1.0, -2.0, -3.0;
  Eigen::VectorXd result3 = ActivationLayer::relu(input3);
  Eigen::VectorXd expected3 = Eigen::VectorXd::Zero(3);

  print_test_result("All negative (zero)",
                    vectors_almost_equal(result3, expected3));
  print_vector("Input", input3);
  print_vector("Result", result3);

  // 测试用例4: 包含零
  Eigen::VectorXd input4(5);
  input4 << -2.0, -1.0, 0.0, 1.0, 2.0;
  Eigen::VectorXd result4 = ActivationLayer::relu(input4);
  Eigen::VectorXd expected4(5);
  expected4 << 0.0, 0.0, 0.0, 1.0, 2.0;

  print_test_result("With zeros", vectors_almost_equal(result4, expected4));
  print_vector("Input", input4);
  print_vector("Result", result4);
}

// 边界值测试
void test_edge_cases() {
  std::cout << "\n=== Testing Edge Cases ===" << std::endl;

  // 空向量测试
  try {
    Eigen::VectorXd empty;
    Eigen::VectorXd result = ActivationLayer::soft_max(empty);
    print_test_result("Empty vector soft_max", result.size() == 0);
  } catch (...) {
    print_test_result("Empty vector soft_max", false);
  }

  try {
    Eigen::VectorXd empty;
    Eigen::VectorXd result = ActivationLayer::relu(empty);
    print_test_result("Empty vector relu", result.size() == 0);
  } catch (...) {
    print_test_result("Empty vector relu", false);
  }

  // 单元素测试
  Eigen::VectorXd single(1);
  single << 5.0;
  Eigen::VectorXd softmax_single = ActivationLayer::soft_max(single);
  print_test_result("Single element soft_max",
                    std::abs(softmax_single[0] - 1.0) < 1e-6);
  std::cout << "Single element softmax result: " << softmax_single[0]
            << std::endl;

  Eigen::VectorXd relu_single = ActivationLayer::relu(single);
  print_test_result("Single element relu",
                    std::abs(relu_single[0] - 5.0) < 1e-6);
}

void ActivationLayer::test() {
  std::cout << "Testing Activation Functions" << std::endl;
  std::cout << "============================" << std::endl;

  test_soft_max();
  test_relu();
  test_edge_cases();

  std::cout << "\n=== Testing Complete ===" << std::endl;
}