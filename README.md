## UAV Disaster Damage Detection Using Semantic Segmentation

### Introduction  
Rapid assessment of structural damage after a hurricane or earthquake is essential to deploy rescue teams and allocate resources.  In this project, we fine-tune a state-of-the-art semantic segmentation network to automatically identify damaged buildings in high-resolution drone images.  By combining pixel-level labels with a cost model, we deliver both a damage map and an estimated dollar value of total loss.

### Dataset  
- **RescueNet**: 4,494 aerial images (3000 × 4000 px) from Hurricane Michael aftermath  
- **Annotations**: 11 classes (undamaged building, minor damage, major damage, total destruction, water, road, tree, vehicle, pool, background)  
- **Split**: 80 % train, 10 % validation, 10 % test  

### Methodology  
1. **Preprocessing**  
   - Convert images to RGB, normalize to pretrained ImageNet statistics  
   - Resize on-the-fly to 500 × 800 in the model’s forward pass for efficient training  
2. **Segmentation Model**  
   - Base network: DeepLabV3+ with ResNet-50 backbone, pretrained on ImageNet  
   - Replace output head to predict 11 classes  
   - Fine-tuning regimes:  
     - Model 1: 3 epochs full network + 4 epochs head  
     - Model 2: 7 epochs head only  
   - Training hardware: Google Colab TPU, batch size = 25  
3. **Cost Estimation**  
   - Translate segmentation pixels to area using 0.01 m² per pixel  
   - Apply per-class cost factors (\$50/m² for minor damage, \$300/m² for major, \$1,200/m² for total destruction)  
   - Sum across all pixels to compute total damage  

### Evaluation  
- **Segmentation Metric**: mean Intersection over Union (mIoU) across 11 classes  
- **Model 1 Performance**:  
  - mIoU = 50.6 % on test set  
  - Visual results accurately capture building outlines and damage severity  
- **Cost Estimation**:  
  - Baseline damage estimate = \$1.25 billion  
  - High sensitivity to segmentation errors (178 % prediction error on a small test subset)  

### Key Takeaways  
- Fine-tuned DeepLabV3+ can identify damage patterns with moderate accuracy after just a few epochs.  
- Economic cost estimates highlight the need for extremely accurate segmentation before relying on dollar-value outputs.  
- Future improvements include longer training schedules, lighter architectures for edge deployment, and region-specific cost factors.

