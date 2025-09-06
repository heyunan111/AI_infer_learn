#include "dataset.h"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

Eigen::MatrixXd DataSet::load_image(const std::string &file_name) {
  cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
  if (image.empty()) {
    std::cerr << "无法读取图像: " << file_name << std::endl;
    return {};
  }
  cv::Mat image_double;
  image.convertTo(image_double, CV_64F);
  Eigen::MatrixXd m;
  cv::cv2eigen(image_double, m);
  return m;
}

Eigen::MatrixXd DataSet::load_opencv_yml_matrix(const std::string &filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Failed to open yml: " + filename);
  }
  cv::Mat mat;
  fs["data"] >> mat;
  fs.release();

  if (mat.empty())
    throw std::runtime_error("Empty mat in file: " + filename);

  // 转换为 double 类型矩阵，方便转换到 Eigen::MatrixXd
  cv::Mat mat_d;
  if (mat.depth() == CV_64F)
    mat.convertTo(mat_d, CV_64F);
  else
    mat.convertTo(mat_d, CV_64F);

  Eigen::MatrixXd out(mat_d.rows, mat_d.cols);
  for (int r = 0; r < mat_d.rows; ++r)
    for (int c = 0; c < mat_d.cols; ++c)
      out(r, c) = mat_d.at<double>(r, c);

  return out;
}

Eigen::VectorXd DataSet::load_bias_as_vector(const std::string &filename,
                                             int expected_size) {
  Eigen::MatrixXd m = load_opencv_yml_matrix(filename);
  Eigen::VectorXd v;
  if (m.cols() == 1 && m.rows() >= 1) {
    v = m.col(0);
  } else if (m.rows() == 1 && m.cols() >= 1) {
    v = m.row(0).transpose();
  } else if (m.size() == expected_size) {
    // 当 m 是一个一维数组，但没有明确行/列时
    v = Eigen::Map<Eigen::VectorXd>(m.data(), m.size());
  } else {
    // 虽然不符合，但尽量尝试按扁平读取
    v = Eigen::Map<Eigen::VectorXd>(m.data(), m.size());
  }

  if (expected_size > 0 && v.size() != expected_size) {
    std::cerr << "Warning: bias size mismatch for " << filename << " expected "
              << expected_size << " got " << v.size() << std::endl;
  }
  return v;
}

Eigen::VectorXd DataSet::prepare_input(const std::string &file_name) {
  Eigen::MatrixXd img = load_image(file_name);
  if (img.rows() != 28 || img.cols() != 28) {
    std::cerr << "Warning: input image size is " << img.rows() << "x"
              << img.cols() << " (expected 28x28). Consider resizing.\n";
  }

  // to [0,1]
  img = img.array() / 255.0;

  // PyTorch Normalize((0.1307,), (0.3081,))
  const double mean = 0.1307;
  const double stdv = 0.3081;
  img = (img.array() - mean) / stdv;

  // 明确使用 "row-major" flatten (PyTorch 的 flatten 是按行)
  const int H = img.rows();
  const int W = img.cols();
  Eigen::VectorXd input(H * W);
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      input[r * W + c] = img(r, c); // row-major ordering
    }
  }
  return input;
}

Eigen::MatrixXd DataSet::load_weight_with_check(const std::string &filename,
                                                int expected_out,
                                                int expected_in) {
  Eigen::MatrixXd W = load_opencv_yml_matrix(filename);
  std::cout << "Loaded " << filename << " shape = " << W.rows() << " x "
            << W.cols() << std::endl;

  if (W.rows() == expected_out && W.cols() == expected_in) {
    // ok
    return W;
  } else if (W.rows() == expected_in && W.cols() == expected_out) {
    std::cerr << "Auto-transposing weights for " << filename
              << " (swapped dims).\n";
    return W.transpose();
  } else if (W.size() == expected_out * expected_in) {
    // 可能是被存成一行/一列，尝试 reshape row-major -> out x in
    std::cerr << "Reshaping flat weight matrix for " << filename << "\n";
    Eigen::MatrixXd Wflat =
        Eigen::Map<Eigen::MatrixXd>(W.data(), expected_out, expected_in);
    return Wflat;
  } else {
    std::cerr << "WARNING: unexpected weight shape for " << filename
              << ", expected (" << expected_out << "x" << expected_in
              << "), got (" << W.rows() << "x" << W.cols()
              << "). Attempting to use as-is.\n";
    return W;
  }
}

// --- debug printing small prefix
void DataSet::print_vector_head(const Eigen::VectorXd &v, int n) {
  int m = std::min<int>(n, v.size());
  for (int i = 0; i < m; ++i)
    std::cout << v[i] << " ";
  std::cout << "\n";
}

Eigen::MatrixXd
DataSet::load_dataset_from_folder(const std::string &folder_path) {
  std::vector<Eigen::VectorXd> samples;

  // 遍历目录
  for (const auto &entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      std::string file_name = entry.path().string();
      Eigen::VectorXd vec = prepare_input(file_name);

      if (vec.size() > 0) {
        samples.push_back(vec);
      }
    }
  }

  if (samples.empty()) {
    throw std::runtime_error("No valid images found in folder: " + folder_path);
  }

  // 假设所有样本向量长度相同
  size_t feature_size = samples[0].size();
  size_t num_samples = samples.size();

  // 创建大矩阵: 行 = 样本数，列 = 特征维度
  Eigen::MatrixXd data(num_samples, feature_size);
  for (size_t i = 0; i < num_samples; ++i) {
    if (samples[i].size() != feature_size) {
      throw std::runtime_error("Inconsistent feature size in input images!");
    }
    data.row(i) = samples[i];
  }

  return data;
}

void DataSet::load_data_set(const std::string &imageFloder,
                            const std::string &labsText) {
  auto txt = load_labs_from_txt(labsText);
  for (int i = 0; i < txt.size(); ++i) {
    MNISetData data;
    data.data =
        prepare_input(imageFloder + std::to_string(txt[i].first) + ".png");
    data.lab = txt[i].second;
    _dataSet.push_back(data);
  }
}

std::vector<std::pair<int, int>>
DataSet::load_labs_from_txt(const std::string &file_path) {
  std::vector<std::pair<int, int>> result;
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + file_path);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty())
      continue;

    std::istringstream iss(line);
    int first, second;
    if (iss >> first >> second) {
      result.emplace_back(first, second);
    } else {
      std::cerr << "Warning: failed to parse line: " << line << std::endl;
    }
  }

  return result;
}