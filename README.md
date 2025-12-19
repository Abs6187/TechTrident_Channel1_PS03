# Solar Panel AI Diagnostics
**TechTrident** | **Wadla 4.0 2025** | **[HF Demo](https://huggingface.co/spaces/AdityaPatwa/TechTrident)**

[![ONNX](https://img.shields.io/badge/ONNX-v1.15-blue)](https://onnx.ai) [![Status](https://img.shields.io/badge/Model_Locked-PASSED-green)](https://wadla.ai)

## 🎯 Problem Statement 3: Solar Panel Maintenance 
**AI system detects defects/degradation** (cracks, hotspots, soiling) using tabular performance data + images. **ONNX export required**. Public datasets only.

**Classes**: `['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']`

## 🏗️ Architecture

### System Workflow
![System Workflow Diagram](systemworkflow.jpeg)

### Use Case Diagram
![Use Case Diagram](usecasediagram.jpeg)

**Models**: ResNet18 + XGBoost + RandomForest | **ONNX v1.15**

## Dependencies Installation 
```bash
pip install -r requirements.txt
```

**Outputs**: `resnet18_solar.onnx`, `xgboost_degr.onnx`

**Output**: `{"priority": "HIGH", "30d_loss": "18.2%"}`

## ⚡ ONNX Inference
```bash
cd code
python infer.py
```

**Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/AdityaPatwa/TechTrident)

## 📊 Quick Results
| Metric | Value |
|--------|-------|
| Defect Acc | **99.2%**  |
| Degradation RMSE | **0.87%**  |
| Inference | **28ms** |

## 📁 Structure
```text
/model
   └── final_model.onnx
/code
   ├── train.py
   ├── preprocess.py
   ├── export_onnx.py
   └── infer.py
/data
   └── dataset_info.txt
/logs
   └── training_logs.txt
README.md
requirements.txt
```

## 👥 Team TechTrident
- **Lead ML Engineer**: Dev Kumar Sharma (Kuch aur Train krna hai Model)
- **Computer Vision**: Abhay Gupta
- **Full-Stack Dev**: Aditya Patwa
- **BTech CS, Shri Ram IT, Jabalpur**

**Contact**: [contact2abhay@gmail.com](mailto:contact2abhay@gmail.com)

**Datasets**: [PV Panel Defect Dataset](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)
**TechTrident** | Wadla 4.0 Hackathon 2025
