---

# **PyTorch Dynamic Quantization**

This repository demonstrates **dynamic quantization** in PyTorch, a technique that reduces the precision of model weights to **INT8** during runtime for **faster inference** and **lower memory usage**. Dynamic quantization is particularly useful for models like BERT, where the activations remain in floating-point precision, but the weights are quantized to INT8.

---

## **Key Features**
- **Dynamic Quantization**: Convert model weights to INT8 precision during runtime.
- **Reduced Model Size**: Significantly smaller model size compared to FP32.
- **Improved Inference Speed**: Faster inference on CPU with minimal loss in accuracy.
- **BERT Example**: Includes a BERT-based Question Answering (QA) example to demonstrate dynamic quantization.

---

## **Results**

### **Model Sizes**
| Model Precision | Size      |
|-----------------|-----------|
| FP32            | 411.00 MB |
| INT8            | 168.05 MB |

### **BERT QA Example**
- **Text**:  
  According to PolitiFact, the top 400 richest Americans "have more wealth than half of all Americans combined." According to the New York Times on July 22, 2014, the "richest 1 percent in the United States now own more wealth than the bottom 90 percent". Inherited wealth may help explain why many Americans who have become rich may have had a "substantial head start". In September 2012, according to the Institute for Policy Studies, "over 60 percent" of the Forbes richest 400 Americans "grew up in substantial privilege".

- **Question**:  
  What publication printed that the wealthiest 1% have more money than those in the bottom 90%?

- **Model Answer**:  
  New York Times

- **Dynamic Quantized Model Answer**:  
  New York Times

### **Inference Latency**
| Model Precision          | Device | Latency (ms / sample) |
|--------------------------|--------|-----------------------|
| FP32                     | CPU    | 52.27                 |
| Dynamic Quantized (INT8) | CPU    | 40.63                 |
| FP32                     | CUDA   | 7.02                  |

---

## **Getting Started**

### **1. Installation**
To get started with PyTorch Dynamic Quantization, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/PyTorch-Dynamic-Quantization.git
   cd PyTorch-Dynamic-Quantization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **2. Run the BERT QA Example**
Run the provided script to perform dynamic quantization and evaluate the BERT QA model:
```bash
python qa.py
```

---

## **Dynamic Quantization Workflow**

### **1. Quantization**
- **Description**: Convert model weights to INT8 precision during runtime.
- **Steps**:
  1. Load the pre-trained FP32 model.
  2. Apply dynamic quantization to the model weights.
  3. Save the quantized model for inference.

### **2. Inference**
- **Description**: Run inference with the quantized model and compare results with the FP32 model.
- **Steps**:
  1. Load the quantized model.
  2. Run inference on the input text and question.
  3. Compare accuracy and latency with the FP32 model.

---

## **Performance Benchmarks**

### **Model Size Comparison**
| Model Precision | Size      |
|-----------------|-----------|
| FP32            | 411.00 MB |
| INT8            | 168.05 MB |

### **Inference Latency Comparison**
| Model Precision          | Device | Latency (ms / sample) |
|--------------------------|--------|-----------------------|
| FP32                     | CPU    | 52.27                 |
| Dynamic Quantized (INT8) | CPU    | 40.63                 |
| FP32                     | CUDA   | 7.02                  |

---

## **Folder Structure**
Here’s an overview of the repository structure:

```
PyTorch-Dynamic-Quantization/
├── qa.py                      # Script for BERT QA with dynamic quantization
├── requirements.txt           # List of dependencies
├── README.md                  # Project README file
└── models/                    # Folder for pre-trained and quantized models
    ├── bert_fp32.pth          # FP32 BERT model
    └── bert_int8.pth          # INT8 BERT model
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
- Special thanks to the **Hugging Face team** for their work on the Transformers library and BERT model.

---
