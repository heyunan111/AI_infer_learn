#include "mlp_network.h"
#include "dense_layer.h"
#include "layer.h"
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <memory>
#include <stdexcept>

// --- 网络构建 ---
void MLPNetwork::addLayer(std::unique_ptr<Layer> layer) {
  _layers.push_back(std::move(layer));
}
bool MLPNetwork::checkConsistency(bool throw_on_error) const {
  if (_layers.empty()) {
    return true;
  }
  int preOutDim = _layers[0]->outputDim();
  for (size_t i = 1; i < _layers.size(); ++i) {
    if (preOutDim != _layers[i]->inputDim()) {
      if (throw_on_error) {
        throw std::runtime_error(
            "MLP NetWork pre-output dimension != net input dimension");
      }
      return false;
    }
  }
  return true;
}

// --- 推理 ---
Eigen::VectorXd MLPNetwork::forward(const Eigen::VectorXd &x) const {
  if (_layers.empty()) {
    throw std::runtime_error("MLP NetWork empty layers");
  }
  Eigen::VectorXd res = x;
  for (size_t i = 0; i < _layers.size(); ++i) {
    res = _layers[i]->compute(res);
  }
  return res;
}

// --- 权重持久化 ---
void MLPNetwork::loadWeights(const std::string &fileName) {}
void MLPNetwork::saveWeights(const std::string &fileName) const {}

// --- 元信息 ---
int MLPNetwork::inputDim() const {
  if (_layers.empty()) {
    throw std::runtime_error("MLP NetWork empty layers");
  }
  return _layers.front()->inputDim();
}
int MLPNetwork::outputDim() const {
  if (_layers.empty()) {
    throw std::runtime_error("MLP NetWork empty layers");
  }
  return _layers.back()->outputDim();
}