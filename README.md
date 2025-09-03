# 📌 AI 系统工程 6 个月学习任务清单

## **第 1 个月：C++ & 数值计算基础**

✅ **第 1 周**

~~复习 C++11/14 语法（智能指针、Move 语义）~~

~~学习 RAII、异常安全、内存泄漏检测（Valgrind）~~

✅ **第 2 周**

~~学习 STL 容器 & 算法（vector、map、thread、future）~~

~~写一个 **线程池**，支持提交任务并返回结果~~

✅ **第 3 周**

* [X]  学习矩阵运算（矩阵乘法、转置、点积）
* [X]  用 Eigen 实现矩阵乘法
* [ ]  写一个 **Softmax + ReLU** 函数

✅ **第 4 周**

* [ ]  用 C++ 写一个 **MLP 前向推理框架**（2层全连接 + ReLU + Softmax）
* [ ]  输入 MNIST 图片（OpenCV 读入），输出分类结果

---

## **第 2 个月：ONNX Runtime**

✅ **第 5 周**

* [ ]  学习 PyTorch → 导出 ONNX 模型（ResNet18）
* [ ]  学 ONNX 格式结构（Graph、Node、Tensor）

✅ **第 6 周**

* [ ]  学 ONNX Runtime C++ API（加载模型、推理）
* [ ]  在 C++ 里运行 ResNet18 推理一张图片

✅ **第 7 周**

* [ ]  写一个 CLI 工具：`./classifier input.jpg` → 输出 top1 类别
* [ ]  增加 **批量推理**（一次输入多张图片）

✅ **第 8 周**

* [ ]  学 gRPC 基础
* [ ]  写一个小 **推理服务**（客户端发图片 → 服务端返回预测）

---

## **第 3 个月：CUDA 基础**

✅ **第 9 周**

* [ ]  学 CUDA 编程模型（grid、block、thread）
* [ ]  写第一个 CUDA 程序（向量加法）

✅ **第 10 周**

* [ ]  写一个 **CUDA 矩阵乘法**
* [ ]  对比 CPU vs CUDA 性能

✅ **第 11 周**

* [ ]  学共享内存、寄存器优化
* [ ]  优化 CUDA 矩阵乘法（tiling）

✅ **第 12 周**

* [ ]  把第 1 个月的 MLP **移植到 CUDA**
* [ ]  比较 CPU / Eigen / CUDA 性能

---

## **第 4 个月：TensorRT**

✅ **第 13 周**

* [ ]  安装 TensorRT & 学 Builder API
* [ ]  把 ResNet18 转换成 TensorRT Engine

✅ **第 14 周**

* [ ]  跑通 TensorRT 推理 ResNet18
* [ ]  统计推理时间（FP32）

✅ **第 15 周**

* [ ]  学习 FP16/INT8 量化
* [ ]  对比 FP32 / FP16 / INT8 的速度和精度

✅ **第 16 周**

* [ ]  用 TensorRT 部署 YOLOv5
* [ ]  写个 Demo：输入图片 → 输出检测框

---

## **第 5 个月：算子开发**

✅ **第 17 周**

* [ ]  学卷积（Conv2D 数学公式）
* [ ]  写一个 CPU 版 Conv2D（暴力实现）

✅ **第 18 周**

* [ ]  优化 Conv2D（im2col + 矩阵乘法）
* [ ]  比较暴力实现 vs 优化性能

✅ **第 19 周**

* [ ]  写 CUDA 版 Conv2D（基础实现）
* [ ]  对比 cuDNN Conv 性能

✅ **第 20 周**

* [ ]  学 LayerNorm 公式
* [ ]  写 CPU + CUDA 版 LayerNorm

---

## **第 6 个月：工程化**

✅ **第 21 周**

* [ ]  学 gRPC 进阶（双向流）
* [ ]  写推理服务：客户端发送图片流，服务端返回分类结果流

✅ **第 22 周**

* [ ]  学 Docker 基础（build/run）
* [ ]  把推理服务打包成 Docker 镜像

✅ **第 23 周**

* [ ]  在 Linux 服务器上部署 Docker 容器
* [ ]  压测服务 QPS

✅ **第 24 周**

* [ ]  学性能分析工具：`perf`、`nsight`
* [ ]  调优：减少延迟、提升吞吐

---

# 📦 最终交付物

1. **C++ MLP 前向推理框架（CPU + CUDA）**
2. **ONNX Runtime + TensorRT 部署 ResNet / YOLO**
3. **自研卷积 & LayerNorm 算子（CPU + CUDA 版）**
4. **完整 AI 推理服务（gRPC + Docker 化部署）**
