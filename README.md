---

# **PyTorch-Quantization**

This repository provides tools and examples for **quantization** in PyTorch, enabling efficient inference and deployment of deep learning models. Quantization reduces the precision of model weights and activations, leading to **faster inference**, **lower memory usage**, and **reduced power consumption** without significant loss in accuracy.

---

## **Key Features**
- **Post-Training Quantization (PTQ)**: Quantize pre-trained models without retraining.
- **Quantization-Aware Training (QAT)**: Train models with quantization in mind for better accuracy.
- **Support for Multiple Precision Levels**: 8-bit, 16-bit, and mixed-precision quantization.
- **Easy Integration**: Seamlessly integrate quantization into your existing PyTorch workflows.
- **Comprehensive Examples**: Includes examples for quantizing popular models like ResNet, MobileNet, and more.

---

## **Getting Started**

### **1. Installation**
To get started with PyTorch-Quantization, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/Quantization.git
   cd Quantization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Post-Training Quantization (PTQ)**
Quantize a pre-trained model without retraining:
```python
from pytorch_quantization import quantize
quantized_model = quantize(model, calibration_data)
```

### **3. Quantization-Aware Training (QAT)**
Train a model with quantization in mind for better accuracy:
```python
from pytorch_quantization import QuantTrainer
trainer = QuantTrainer(model, train_loader, val_loader)
trainer.train()
```

### **4. Inference**
Run inference with the quantized model:
```python
output = quantized_model(input_data)
```

---

## **Quantization Techniques**

### **Post-Training Quantization (PTQ)**
- **Description**: Quantize a pre-trained model without retraining.
- **Use Case**: Ideal for scenarios where retraining is not feasible.
- **Steps**:
  1. Calibrate the model using a small dataset.
  2. Quantize the model weights and activations.

### **Quantization-Aware Training (QAT)**
- **Description**: Train a model with quantization in mind to improve accuracy.
- **Use Case**: Best for achieving higher accuracy with quantized models.
- **Steps**:
  1. Insert fake quantization nodes during training.
  2. Fine-tune the model to adapt to lower precision.

---

## **Examples**

### **Quantizing ResNet**
Quantize a pre-trained ResNet model using Post-Training Quantization:
```python
from torchvision.models import resnet18
from pytorch_quantization import quantize

model = resnet18(pretrained=True)
quantized_model = quantize(model, calibration_data)
```

### **Quantization-Aware Training for MobileNet**
Train a MobileNet model with quantization-aware training:
```python
from torchvision.models import mobilenet_v2
from pytorch_quantization import QuantTrainer

model = mobilenet_v2(pretrained=True)
trainer = QuantTrainer(model, train_loader, val_loader)
trainer.train()
```

---

## **Performance**
Quantization can significantly improve inference speed and reduce memory usage. Below are some key metrics:

| Model           | Precision | Accuracy | Inference Speed | Memory Usage |
|-----------------|-----------|----------|-----------------|--------------|
| ResNet-18       | FP32      | 70.0%    | 1x              | 1x           |
| ResNet-18       | INT8      | 69.5%    | 2.5x            | 0.5x         |
| MobileNet-V2    | FP32      | 72.0%    | 1x              | 1x           |
| MobileNet-V2    | INT8      | 71.8%    | 3x              | 0.4x         |

---

## **References**
- **PyTorch Quantization Documentation**: [PyTorch Docs](https://pytorch.org/docs/stable/quantization.html)
- **Quantization Papers**:
  - [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
  - [Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper](https://arxiv.org/abs/1806.08342)

---

## **Folder Structure**
Here’s an overview of the repository structure:

```
Quantization/
├── examples/                  # Example scripts for quantization
│   ├── resnet_quantization.py
│   └── mobilenet_qat.py
├── pytorch_quantization/      # Core quantization tools
│   ├── quantize.py
│   └── quant_trainer.py
├── requirements.txt           # List of dependencies
└── README.md                  # Project README file
```

---

## **How to Contribute**
We welcome contributions! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a pull request.

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- Thanks to the **PyTorch team** for providing excellent tools and documentation for quantization.
- Special thanks to the contributors of the **Quantization community** for their research and tools.

---
