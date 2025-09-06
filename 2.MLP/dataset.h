#pragma once

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

struct MNISetData {
  Eigen::VectorXd data;
  int lab;
};

class DataSet {
public:
  static Eigen::VectorXd prepare_input(const std::string &file_name);
  static Eigen::MatrixXd load_opencv_yml_matrix(const std::string &filename);
  static Eigen::VectorXd load_bias_as_vector(const std::string &filename,
                                             int expected_size = -1);
  static Eigen::MatrixXd load_weight_with_check(const std::string &filename,
                                                int expected_out,
                                                int expected_in);
  static void print_vector_head(const Eigen::VectorXd &v, int n = 8);
  static Eigen::MatrixXd
  load_dataset_from_folder(const std::string &folder_path);

  void load_data_set(const std::string &imageFloder,
                     const std::string &labsText);
  std::vector<MNISetData> getDataSet() { return _dataSet; }

private:
  std::vector<std::pair<int, int>>
  load_labs_from_txt(const std::string &file_path);
  static Eigen::MatrixXd load_image(const std::string &file_name);

  std::vector<MNISetData> _dataSet;
};
