
# PyTorch Dynamic Quantization

- python qa.py 

- Model Sizes

 ===========================================================================
- FP32 Model Size: 411.00 MB
- INT8 Model Size: 168.05 MB

  ===========================================================================
- BERT QA Example

  ===========================================================================

- Text: 
- According to PolitiFact the top 400 richest Americans "have more wealth than half of all Americans combined." According to the New York Times on July 22, 2014, the "richest 1 percent in the United States now own more wealth than the bottom 90 percent". Inherited wealth may help explain why many Americans who have become rich may have had a "substantial head start". In September 2012, according to the Institute for Policy Studies, "over 60 percent" of the Forbes richest 400 Americans "grew up in substantial privilege".
- Question: 
- What publication printed that the wealthiest 1% have more money than those in the bottom 90%?
- Model Answer: 
- New York Times
- Dynamic Quantized Model Answer: 
- New York Times
 
 ===========================================================================
- BERT QA Inference Latencies
- 
 ============================================================================

- CPU Inference Latency: 52.27 ms / sample
- Dynamic Quantized CPU Inference Latency: 40.63 ms / sample
- CUDA Inference Latency: 7.02 ms / sample


