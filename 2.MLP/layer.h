#pragma once

#include <Eigen/Dense>

class Layer {
public:
  virtual ~Layer() = default;
  virtual Eigen::VectorXd compute(const Eigen::VectorXd &x) = 0;
  virtual int inputDim() const = 0;
  virtual int outputDim() const = 0;
};