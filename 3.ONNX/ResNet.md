```planuml
@startuml
skinparam rectangle {
  BackgroundColor White
  BorderColor Black
  RoundCorner 10
}

title ResNet-18 Forward Pass (All Blocks)

' === Stem ===
rectangle "Input\n[3x224x224]" as input
rectangle "Conv7x7,s2\nBN+ReLU\n[64x112x112]" as conv1
rectangle "MaxPool3x3,s2\n[64x56x56]" as pool

input --> conv1 --> pool

' === Layer1 ===
rectangle "Layer1 Block1\nBasicBlock\nin: [64x56x56]\nout: [64x56x56]" as l1b1
rectangle "Layer1 Block2\nBasicBlock\nin: [64x56x56]\nout: [64x56x56]" as l1b2

pool --> l1b1 --> l1b2

' === Layer2 ===
rectangle "Layer2 Block1\nBasicBlock (s2)\nshortcut 1x1\nin: [64x56x56]\nout: [128x28x28]" as l2b1
rectangle "Layer2 Block2\nBasicBlock\nin: [128x28x28]\nout: [128x28x28]" as l2b2

l1b2 --> l2b1 --> l2b2

' === Layer3 ===
rectangle "Layer3 Block1\nBasicBlock (s2)\nshortcut 1x1\nin: [128x28x28]\nout: [256x14x14]" as l3b1
rectangle "Layer3 Block2\nBasicBlock\nin: [256x14x14]\nout: [256x14x14]" as l3b2

l2b2 --> l3b1 --> l3b2

' === Layer4 ===
rectangle "Layer4 Block1\nBasicBlock (s2)\nshortcut 1x1\nin: [256x14x14]\nout: [512x7x7]" as l4b1
rectangle "Layer4 Block2\nBasicBlock\nin: [512x7x7]\nout: [512x7x7]" as l4b2

l3b2 --> l4b1 --> l4b2

' === Head ===
rectangle "Global AvgPool\n[512x1x1]" as gap
rectangle "FC + Softmax\n[#classes]" as fc

l4b2 --> gap --> fc
@enduml

```





非常好，这一步我们就来 **用一个具体输入 tensor 手推一遍 ResNet-18 的推理（forward/infer）过程**，把 **shape 和数据流** 全部走一遍，这样你能直观看到数据在网络里是怎么流动和变化的。

---

## 📥 假设输入

我们设定输入是一张标准 ImageNet 尺寸图像：

* **输入 x0x\_0**:
  x0∈R1×3×224×224x\_0 \\in \\mathbb{R}^{1\\times 3\\times 224\\times 224}
  * batch size = 1
  * channel = 3 (RGB)
  * height × width = 224 × 224
  * 值范围 [0,1]，假设已归一化

---

## 🧮 ResNet-18 推理流程（shape 推导）

---

### 🟢 stem 部分

**Conv7x7, stride=2, padding=3, out=64**

out\_shape=224+2⋅3−72+1=112\\text{out\\\_shape} = \\frac{224+2\\cdot3-7}{2}+1 = 112* conv: [1,3,224,224]→[1,64,112,112][1,3,224,224] \\to [1,64,112,112]

* bn/relu: 不变
* maxpool3x3, stride=2 → [1,64,56,56][1,64,56,56]

✅ 得到：

x1∈R1×64×56×56x\_1 \\in \\mathbb{R}^{1\\times 64\\times 56\\times 56}---

### 🟢 Layer1（2个BasicBlock, stride=1）

* Block1:
  * conv3x3: [1,64,56,56]→[1,64,56,56][1,64,56,56]\\to[1,64,56,56]
  * conv3x3: [1,64,56,56]→[1,64,56,56][1,64,56,56]\\to[1,64,56,56]
  * shortcut: identity
  * 输出 shape: [1,64,56,56][1,64,56,56]
* Block2: 同上

✅ 得到：

x2∈R1×64×56×56x\_2 \\in \\mathbb{R}^{1\\times 64\\times 56\\times 56}---

### 🟢 Layer2（2个BasicBlock，第1个 stride=2）

* Block1:
  * conv3x3, s=2: [1,64,56,56]→[1,128,28,28][1,64,56,56]\\to[1,128,28,28]
  * conv3x3: [1,128,28,28]→[1,128,28,28][1,128,28,28]\\to[1,128,28,28]
  * shortcut: 1x1 conv s=2 [1,64,56,56]→[1,128,28,28][1,64,56,56]\\to[1,128,28,28]
  * 残差相加后 relu
* Block2:
  * 输入 [1,128,28,28][1,128,28,28]，输出 [1,128,28,28][1,128,28,28]

✅ 得到：

x3∈R1×128×28×28x\_3 \\in \\mathbb{R}^{1\\times 128\\times 28\\times 28}---

### 🟢 Layer3（2个BasicBlock，第1个 stride=2）

* Block1: [1,128,28,28]→[1,256,14,14][1,128,28,28] \\to [1,256,14,14] + shortcut 1x1
* Block2: [1,256,14,14]→[1,256,14,14][1,256,14,14] \\to [1,256,14,14]

✅ 得到：

x4∈R1×256×14×14x\_4 \\in \\mathbb{R}^{1\\times 256\\times 14\\times 14}---

### 🟢 Layer4（2个BasicBlock，第1个 stride=2）

* Block1: [1,256,14,14]→[1,512,7,7][1,256,14,14] \\to [1,512,7,7] + shortcut 1x1
* Block2: [1,512,7,7]→[1,512,7,7][1,512,7,7] \\to [1,512,7,7]

✅ 得到：

x5∈R1×512×7×7x\_5 \\in \\mathbb{R}^{1\\times 512\\times 7\\times 7}---

### 🟢 分类头

* GlobalAvgPool: [1,512,7,7]→[1,512,1,1][1,512,7,7]\\to[1,512,1,1]
* Flatten: [1,512][1,512]
* Fully Connected: [1,512]→[1,1000][1,512]\\to[1,1000]
* Softmax: [1,1000][1,1000]（概率分布）

✅ 最终输出：

y^∈R1×1000\\hat{y} \\in \\mathbb{R}^{1\\times 1000}---

## 📝 总结表格


| 层         | 输出 shape        |
| ---------- | ----------------- |
| Input      | (1, 3, 224, 224)  |
| Conv7x7+s2 | (1, 64, 112, 112) |
| MaxPool    | (1, 64, 56, 56)   |
| Layer1     | (1, 64, 56, 56)   |
| Layer2     | (1, 128, 28, 28)  |
| Layer3     | (1, 256, 14, 14)  |
| Layer4     | (1, 512, 7, 7)    |
| AvgPool    | (1, 512, 1, 1)    |
| FC         | (1, 1000)         |

---

如果你愿意，我可以**画出这个推理流程的 shape 流程图（用PlantUML，节点是具体shape）**，这样你可以完全照着图手推 shape。

要不要我把这个 shape 流程也画出来？（和你上一步的结构图连起来）
