# ENDG-511-Final-Project

## EEG-Based Insomnia Detection Using Deep Learning on sleep EDF dataset 

## Overview
This project focuses on classifying EEG signals for insomnia detection using a hybrid deep learning model. The raw data was sourced from polysomnography (PSG) recordings, which include multiple biosignals (EEG, EOG, EMG, etc.). Only EEG channels were extracted, preprocessed, and saved as .npy files in two folders: normal and insomnia.

## Problem Statement
Manual interpretation of PSG data is time-consuming and resource-intensive (Iber et al., 2007). EEG-based analysis offers a more scalable solution, especially when paired with deep learning. Prior work (Casson, 2019; Biswal et al., 2018; Arnulf, 2012) has shown machine learning's potential in sleep classification and disorder detection. This study leverages EEG signals to build an efficient model for automated insomnia detection.

## Dataset & Preprocessing

- Dataset: Sleep-EDF Expanded Database.
- Sampling Rate: 100 Hz.
- Processing:
    - EEG signals extracted from PSG.
    - Signals stored in .npy format under normal/ and insomnia/ directories.

## Feature Extraction

**a. Spectrogram Branch (CNN):**

- Spectrograms generated using STFT.
- Resized to 224×224 RGB images.
- Passed through MobileNetV2 (pre-trained on ImageNet) with classification head removed.

**b. Band Power Branch (MLP):**

- PSD calculated using Welch’s method.
- Mean band power extracted for delta, theta, alpha, and beta bands.
- Processed through a simple MLP.

**c. Fusion and Classification:**

- Outputs from CNN and MLP are concatenated.
- Fully connected layers perform binary classification (normal vs. insomnia).

## Model Architecture

- Built in PyTorch.
- Combines MobileNetV2 (for spectrograms) and MLP (for band power).
- Final classifier trained using binary cross-entropy loss and Adam optimizer.

## Training and Evaluation

- Data split: 72% training, 18% validation, 10% test (stratified).
- Trained for 35 epochs with a learning rate of 1e-4.
- Evaluation metrics:
    - Accuracy/Loss plots
    - Confusion Matrix
    - Precision, Recall, F1-score
    - ROC Curve & AUC

## Structured Pruning

- Method: L1-norm filter pruning on convolutional layers.
- Objective: Reduce model size and inference time while maintaining performance.
- Post-pruning: Fine-tuned model on training set and re-evaluated using original metrics.

## Challenges

- Data Acquisition: Sleep apnea data was excluded due to quality and availability constraints.
- Signal Parsing: Extracting EEG from multi-signal PSG files required custom processing.
- Overfitting: Training beyond 35 epochs degraded validation accuracy; capped epochs.
- Learning Rate Tuning: 1e-4 offered best convergence; higher values were unstable.
- Model Fusion: Careful design needed to combine CNN and MLP effectively.
- Pruning Trade-offs: Filter pruning reduced size with minimal accuracy loss.
- LSTM-CNN Alternative: An LSTM + CNN model was attempted but abandoned due to memory limitations.

## Results

- High test accuracy maintained after pruning.
- Visualization of accuracy curves, ROC, and confusion matrix confirmed stable performance.
- Model compression achieved without significant accuracy loss.

## Conclusion
This hybrid model efficiently classifies EEG signals into normal and insomnia categories using a dual-branch architecture. The integration of spectrogram and band power features enhances accuracy, while structured pruning makes the model suitable for resource-constrained environments. This pipeline has potential applications in real-time sleep monitoring and diagnostic tools.

## References
Casson, A. J. (2019). Wearable EEG and beyond. Biomedical Engineering Letters, 9, 53–71. https://doi.org/10.1007/s13534-018-00093-6

Biswal, S., Sun, H., Goparaju, B., Westover, M. B., Sun, J., & Bianchi, M. T. (2018). Expert-level sleep scoring with deep neural networks. Journal of the American Medical

Arnulf, I. (2012). REM sleep behavior disorder: Motor manifestations and pathophysiology. Movement Disorders, 27(6), 677-689. https://doi.org/10.1002/mds.24957
