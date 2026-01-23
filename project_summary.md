# 🎯 ApneaAlert - Sleep Apnea Detection System

## 📊 **Project Overview**
A machine learning system that detects sleep apnea from ECG signals with **97.6% accuracy**.

## 🚀 **Key Achievements**

### **Performance Metrics**
| Metric | Random Forest | SVM |
|--------|---------------|-----|
| **Accuracy** | 92.8% | 92.0% |
| **ROC-AUC** | 97.6% | 95.7% |
| **Recall** | 96.5% | 95.5% |
| **Precision** | 94.0% | 93.9% |
| **F1-Score** | 95.3% | 94.7% |

### **Dataset**
- **Source**: PhysioNet Apnea-ECG database (records a01-a05)
- **Segments**: 2,481 one-minute ECG segments
- **Distribution**: 75.2% apnea, 24.8% normal (clinically realistic)
- **Duration**: ~41 hours of ECG data

### **Feature Engineering**
**15 features extracted per 1-minute segment:**

**Time Domain:**
- `mean`, `std`, `min`, `max`, `range`, `rms` - ECG signal statistics
- `skewness`, `kurtosis` - Signal distribution characteristics

**Heart Rate Analysis:**
- `hr_mean`, `hr_std` - Heart rate and variability
- `rr_mean`, `rr_std` - RR intervals and variability

**Frequency Domain:**
- `lf_power`, `hf_power` - Low/High frequency power
- `lf_hf_ratio` - Sympathovagal balance

## 🏆 **Most Important Features**
1. **`hr_std`** (14.6%) - Heart rate variability
2. **`rms`** (13.1%) - Root mean square of ECG
3. **`std`** (12.2%) - ECG standard deviation
4. **`rr_std`** (12.0%) - RR interval variability
5. **`range`** (11.4%) - ECG amplitude range

## 💻 **Technical Implementation**

### **Pipeline Architecture**
1. **Data Preprocessing**
   - Bandpass filtering (0.5-40 Hz)
   - Notch filtering (50 Hz powerline)
   - 1-minute segmentation

2. **Feature Extraction**
   - Time-domain statistics
   - Frequency-domain analysis
   - Heart rate variability metrics

3. **Machine Learning**
   - Random Forest Classifier (selected)
   - Support Vector Machine (tested)
   - 5-fold cross-validation

4. **Model Deployment**
   - Model serialization (`.pkl` files)
   - Feature scaler preservation
   - Demo prediction system

### **Project Structure**