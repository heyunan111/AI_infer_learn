#include "dense_layer.h"
#include <Eigen/src/Core/Matrix.h>
DenseLayer::DenseLayer(int input_dim, int output_dim)
    : _input_dimension(input_dim), _output_dimension(output_dim),
      W(Eigen::MatrixXd::Zero(
          output_dim, input_dim)),
      b(Eigen::VectorXd::Zero(output_dim)) {

  // 参数验证
  if (input_dim <= 0 || output_dim <= 0) {
    throw std::invalid_argument("输入和输出维度必须大于0");
  }
}

void DenseLayer::setW(const Eigen::MatrixXd &w) {
  // 维度验证
  if (w.rows() != _output_dimension || w.cols() != _input_dimension) {
    throw std::invalid_argument("权重矩阵维度不匹配");
  }
  this->W = w;
}

void DenseLayer::setB(const Eigen::VectorXd &b) {
  // 维度验证
  if (b.size() != _output_dimension) {
    throw std::invalid_argument("偏置向量维度不匹配");
  }
  this->b = b;
}

Eigen::VectorXd DenseLayer::compute(const Eigen::VectorXd &x) {
  // 输入维度检查
  if (x.size() != _input_dimension) {
    throw std::invalid_argument("输入向量维度不匹配");
  }

  // 计算并返回结果
  return W * x + b;
}