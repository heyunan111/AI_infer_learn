#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

#include "layer.h"

///* `output = W * input + b`
///* `W` 是[输出维度 × 输入维度] 矩阵
// * `b` 是[输出维度] 向量
class DenseLayer : public Layer {
public:
  DenseLayer(int input_dim, int output_dim);

  void setW(const Eigen::MatrixXd &w);
  void setB(const Eigen::VectorXd &b);
  const Eigen::MatrixXd &getW() const { return W; }
  const Eigen::VectorXd &getB() const { return b; }
  int inputDim() const override{ return _input_dimension; }
  int outputDim() const override{ return _output_dimension; }

  /**
   * @brief 计算输出向量 output = W * input + b
   * @param x 输入向量，维度应为 input_dim
   * @return 输出向量，维度为 output_dim
   * @throws std::invalid_argument 如果输入维度不匹配
   */
  Eigen::VectorXd compute(const Eigen::VectorXd &x) override;

private:
  int _input_dimension = 0;
  int _output_dimension = 0;
  Eigen::MatrixXd W;
  Eigen::VectorXd b;
};