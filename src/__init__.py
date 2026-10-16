"""
AI-Powered Medical Image Analysis
=================================
A modular, industry-style pipeline for detecting pneumonia from chest X-rays.

Modules
-------
simulate_data : Virtual simulation dataset generator (synthetic X-ray phantoms)
data_loader   : Load train / validation / test sets from folder structure
preprocessing : Resize, normalization, CLAHE contrast enhancement
model         : CNN and transfer-learning (MobileNetV2) model builders
train         : Training loop with checkpoints, logging and curves
evaluate      : Accuracy, precision, recall, F1, confusion matrix, ROC-AUC
predict       : Single-image inference + Grad-CAM heatmap explanation
visualize     : Publication-style plots for reports and README
"""

__version__ = "1.0.0"
__author__ = "Student Project - AI Medical Image Analysis"
