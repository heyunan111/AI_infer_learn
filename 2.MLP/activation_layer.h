#pragma once

#include "layer.h"

class ActivationLayer : public Layer {
public:
  enum class enActiveFuncType {
    enSoftMax,
    enReLU,
  };
  int inputDim() const override { return _input_dimension; }
  int outputDim() const override { return _output_dimension; }
  ActivationLayer(enActiveFuncType type, int inputDim, int outputDim);
  Eigen::VectorXd compute(const Eigen::VectorXd &x) override;
  static void test();
  static Eigen::VectorXd soft_max(const Eigen::VectorXd &x);
  static Eigen::VectorXd relu(const Eigen::VectorXd &x);

private:
  enActiveFuncType _type;
  int _input_dimension;
  int _output_dimension;
};