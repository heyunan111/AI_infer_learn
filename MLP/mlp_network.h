#pragma once

#include "activation_layer.h"
#include "dense_layer.h"
#include "layer.h"
#include <Eigen/src/Core/Matrix.h>
#include <memory>
#include <vector>

class MLPNetwork {
public:
  // --- 网络构建 ---
  void addLayer(std::unique_ptr<Layer> layer);
  bool checkConsistency(bool throw_on_error = true) const;

  // --- 推理 ---
  Eigen::VectorXd forward(const Eigen::VectorXd &x) const;

  // --- 权重持久化 ---
  void loadWeights(const std::string &fileName);
  void saveWeights(const std::string &fileName) const;

  // --- 元信息 ---
  int inputDim() const;
  int outputDim() const;
  bool empty() const { return _layers.empty(); }

private:
  std::vector<std::unique_ptr<Layer>> _layers;
};