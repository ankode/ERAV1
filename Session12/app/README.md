---
title: CustomResnetModel
emoji: 🐢
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 3.39.0
app_file: app.py
pinned: false
license: mit
---
# Image Classifier App with CustomResNet, Gradio, and PyTorch Lightning

This repository demonstrates the integration of a CustomResNet model with Gradio and PyTorch Lightning, creating an interactive web application for image classification. The app features GradCAM image visualization, misclassified image exploration, image upload with prediction, and displays top predicted classes. 

## Summary

### Model Training and Lightning Integration

The CustomResNet model is retrained using PyTorch Lightning, enhancing training efficiency and organization. Training logs and loss function graphs are available in the [GitHub repository](https://github.com/ankode/ERAV1/tree/main/Session12).

### Gradio Web App

The Gradio-powered web app offers:

1. **GradCAM Images**: Visualize GradCAM outputs highlighting prediction-contributing areas. Customize image count, opacity, and target layer.

2. **Misclassified Images**: Explore misclassified images with adjustable count.

3. **Upload and Prediction**: Upload images for classification. Top predicted classes (up to 10) are displayed.

4. **Example Images**: Quickly test model predictions with 10 included CIFAR-10 dataset images.

## Links

- [GitHub Repository](https://github.com/ankode/ERAV1/tree/main/Session12)
- [Spaces App Link](https://huggingface.co/spaces/ankoder/CustomResnetModel)
- [Spaces README Link](https://huggingface.co/spaces/ankoder/CustomResnetModel/blob/main/README.md)
