#include "activation_layer.h"
#include "dataset.h"
#include "dense_layer.h"
#include "mlp_network.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

MLPNetwork build_mnist_mlp(const std::string &weight_dir) {
  MLPNetwork net;

  // --- Layer 1: Dense(784→256) + ReLU ---
  {
    int in = 784, out = 256;
    auto dense = std::make_unique<DenseLayer>(in, out);

    Eigen::MatrixXd W = DataSet::load_weight_with_check(
        weight_dir + "/fc1_weight.yml", out, in);
    Eigen::VectorXd b =
        DataSet::load_bias_as_vector(weight_dir + "/fc1_bias.yml", out);

    dense->setW(W);
    dense->setB(b);
    net.addLayer(std::move(dense));
    net.addLayer(std::make_unique<ActivationLayer>(
        ActivationLayer::enActiveFuncType::enReLU, 256, 256));
  }

  // --- Layer 2: Dense(256→128) + ReLU ---
  {
    int in = 256, out = 128;
    auto dense = std::make_unique<DenseLayer>(in, out);

    Eigen::MatrixXd W = DataSet::load_weight_with_check(
        weight_dir + "/fc2_weight.yml", out, in);
    Eigen::VectorXd b =
        DataSet::load_bias_as_vector(weight_dir + "/fc2_bias.yml", out);

    dense->setW(W);
    dense->setB(b);
    net.addLayer(std::move(dense));
    net.addLayer(std::make_unique<ActivationLayer>(
        ActivationLayer::enActiveFuncType::enReLU, 128, 128));
  }

  // --- Layer 3: Dense(128→10) ---
  {
    int in = 128, out = 10;
    auto dense = std::make_unique<DenseLayer>(in, out);

    Eigen::MatrixXd W = DataSet::load_weight_with_check(
        weight_dir + "/fc3_weight.yml", out, in);
    Eigen::VectorXd b =
        DataSet::load_bias_as_vector(weight_dir + "/fc3_bias.yml", out);

    dense->setW(W);
    dense->setB(b);

    net.addLayer(std::move(dense));
  }

  // --- 最后一层 Softmax (推理用，可选) ---
  net.addLayer(std::make_unique<ActivationLayer>(
      ActivationLayer::enActiveFuncType::enSoftMax, 10, 10));

  return net;
}

int main() {
  auto mlp =
      build_mnist_mlp("D:/projects/AI_infer_learn/MLP/train/weights_yml");

  Eigen::MatrixXd all_data = DataSet::load_dataset_from_folder(
      "D:/MNIST数据集/mnist_dataset/mnist_dataset/test/test");

  int okNum = 0;
  int errNum = 0;
  DataSet dataset;
  dataset.load_data_set(
      "D:/MNIST数据集/mnist_dataset/mnist_dataset/test/test/",
      "D:/MNIST数据集/mnist_dataset/mnist_dataset/test_labs.txt");
  auto data = dataset.getDataSet();
  for (const auto &imageData : data) {
    Eigen::VectorXd output = mlp.forward(imageData.data);
    int pred = -1;
    output.maxCoeff(&pred);
    if (pred == imageData.lab) {
      okNum++;
    } else {
      errNum++;
    }
  }

  std::cout << "ok num = " << okNum << std::endl;
  std::cout << "err num = " << errNum << std::endl;
  std::cout << "准确率 = "
            << static_cast<double>(okNum) / static_cast<double>(okNum + errNum);
}
