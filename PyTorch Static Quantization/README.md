---

# **PyTorch Static Quantization**

This repository demonstrates **static quantization** in PyTorch, a technique that reduces the precision of model weights and activations to **INT8** for **faster inference** and **lower memory usage** without significant loss in accuracy. Static quantization is particularly useful for deploying models on resource-constrained devices like mobile phones and edge devices.

---

## **Key Features**
- **Static Quantization**: Convert models to INT8 precision for efficient inference.
- **High Accuracy**: Minimal loss in accuracy compared to FP32 models.
- **Performance Benchmarks**: Includes latency and accuracy comparisons for FP32 and INT8 models.
- **Easy-to-Use**: Simple scripts to quantize and evaluate models.

---

## **Results**

### **Accuracy**
- **FP32 Evaluation Accuracy**: 0.869
- **INT8 Evaluation Accuracy**: 0.868

### **Inference Latency**
- **FP32 CPU Inference Latency**: 4.68 ms / sample
- **FP32 CUDA Inference Latency**: 3.70 ms / sample
- **INT8 CPU Inference Latency**: 2.03 ms / sample
- **INT8 JIT CPU Inference Latency**: 0.45 ms / sample

---

## **Getting Started**

### **1. Installation**
To get started with PyTorch Static Quantization, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/PyTorch-Static-Quantization.git
   cd PyTorch-Static-Quantization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Quantize a Model**
Use the provided script to quantize a pre-trained model:
```bash
python quantize_model.py --model_path model.pth --calibration_data calibration_data.pt
```

### **3. Evaluate the Quantized Model**
Evaluate the quantized model's accuracy and latency:
```bash
python evaluate_model.py --model_path quantized_model.pth --test_data test_data.pt
```

---

## **Quantization Workflow**

### **1. Calibration**
- **Description**: Calibrate the model using a small dataset to determine the optimal quantization parameters.
- **Steps**:
  1. Run the model on the calibration dataset.
  2. Collect statistics on activations to determine quantization scales and zero points.

### **2. Quantization**
- **Description**: Convert the model to INT8 precision using the calibration data.
- **Steps**:
  1. Apply quantization to the model weights and activations.
  2. Save the quantized model for inference.

### **3. Evaluation**
- **Description**: Evaluate the quantized model's accuracy and inference latency.
- **Steps**:
  1. Run inference on the test dataset.
  2. Compare accuracy and latency with the FP32 model.

---

## **Performance Benchmarks**

### **Accuracy Comparison**
| Model Precision | Accuracy |
|-----------------|----------|
| FP32            | 0.869    |
| INT8            | 0.868    |

### **Latency Comparison**
| Model Precision | Device | Latency (ms / sample) |
|-----------------|--------|-----------------------|
| FP32            | CPU    | 4.68                  |
| FP32            | CUDA   | 3.70                  |
| INT8            | CPU    | 2.03                  |
| INT8 (JIT)      | CPU    | 0.45                  |

---

## **Folder Structure**
Here’s an overview of the repository structure:

```
PyTorch-Static-Quantization/
├── quantize_model.py          # Script to quantize a model
├── evaluate_model.py          # Script to evaluate the quantized model
├── requirements.txt           # List of dependencies
├── README.md                  # Project README file
└── data/                      # Folder for calibration and test data
    ├── calibration_data.pt    # Calibration dataset
    └── test_data.pt           # Test dataset
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
