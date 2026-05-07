# 🧠 CNN-Based Semantic Edge Detector

A deep learning project that classifies **12 types of edges** in image patches using a custom CNN trained on synthetic and natural image data. Built as a resume project demonstrating a complete 5-stage ML pipeline — from data generation to real-image evaluation.

---

## 📌 Project Overview

Traditional edge detectors like Canny and Sobel only detect *where* edges are. This project goes a step further — it detects **what kind of edge** exists at each location, classifying patches into 12 semantic categories such as horizontal, vertical, diagonal, corner, T-junction, cross, and no-edge.

### Edge Classes (12 Total)
| # | Edge Type |
|---|-----------|
| 0 | Horizontal |
| 1 | Vertical |
| 2 | Diagonal (↗) |
| 3 | Diagonal (↘) |
| 4 | Corner Top-Left |
| 5 | Corner Top-Right |
| 6 | Corner Bottom-Left |
| 7 | Corner Bottom-Right |
| 8 | T-Top |
| 9 | T-Bottom |
| 10 | Cross |
| 11 | No Edge |

---

## 🗂️ Project Structure

```
Edge_Detection_DL/
│
├── 1_EdgeTypes_and_DataGeneration.ipynb   # Define edge types + generate synthetic data
├── 2_Training_CNN.ipynb                   # Train CNN v1 on synthetic data
├── 3_Training_Natural.ipynb               # Fine-tune CNN v2 on natural-like images
├── 4_Testing_and_Tuning.ipynb             # Compare v1 vs v2 vs v3, architecture search
├── 5_Final_Evaluation_Real_Images.ipynb   # Sliding-window inference + metrics
│
├── X_synthetic.npy                        # Synthetic training patches
├── y_synthetic.npy                        # Synthetic labels
├── X_test_synthetic.npy                   # Held-out test patches
├── y_test_synthetic.npy                   # Held-out test labels
│
├── edge_cnn_v1_final.keras                # Model trained on synthetic data
├── edge_cnn_v2_final.keras                # Fine-tuned on natural images
├── edge_cnn_v3_final.keras                # Best model after architecture tuning
│
├── results/
│   └── final_metrics.json                 # F1 / Precision / Recall scores
│
└── requirements.txt
```

> **Note:** Each notebook depends on the artifacts saved by the previous one. Run them in order: NB1 → NB2 → NB3 → NB4 → NB5.

---

## 🔄 Pipeline

```
NB1: Data Generation
   └─ 12,000 synthetic 5×5 patches with Gaussian + salt-pepper noise
         ↓
NB2: CNN v1 Training
   └─ 2× Conv2D + BatchNorm + Dense + Dropout on synthetic data
         ↓
NB3: Fine-tuning on Natural Images (CNN v2)
   └─ 30 procedurally generated images, Canny/Sobel pseudo-labels
   └─ Freeze Conv Block 1, lr=1e-4, 20% synthetic mix
         ↓
NB4: Architecture Testing + Best Model (CNN v3)
   └─ Noise robustness sweep (σ = 0 to 0.5)
   └─ Wider / Deeper / Slim config comparison
         ↓
NB5: Final Evaluation on Real Images
   └─ Sliding-window on Geometric, Complex, Noisy, Fine-line images
   └─ CNN v3 vs Canny vs Sobel comparison
   └─ Edge type heatmaps + confidence maps + F1/Precision/Recall
```

---

## 🏗️ Model Architecture (CNN v1 Base)

```
Input: (5, 5, 1) grayscale patch

Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool
Conv2D(64, 3×3) → BatchNorm → ReLU
Flatten
Dense(128) → Dropout(0.4)
Dense(12) → Softmax
```

---

## 📊 Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| CNN v1 | 98.62% | Trained on synthetic data only |
| CNN v2 | 98.52% | Fine-tuned on natural-like images |
| CNN v3 | 99.17% | Best architecture config |
| Sobel (baseline) | — | Used as reference |
| Canny (baseline) | — | Used as F1 reference |


---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Ankit0431/Edge_Detection_DL.git
cd Edge_Detection_DL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```
Open and run each notebook from `1_EdgeTypes_and_DataGeneration.ipynb` through `5_Final_Evaluation_Real_Images.ipynb`.

---

## 🧰 Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — CNN model building and training
- **OpenCV** — Image processing, Canny/Sobel edge detection
- **NumPy** — Array operations and data handling
- **Matplotlib / Seaborn** — Visualization and heatmaps
- **scikit-learn** — Metrics (F1, Precision, Recall, confusion matrix)

---

## 💡 Key Concepts Demonstrated

- Custom synthetic dataset generation with noise augmentation
- CNN design for small patch classification
- Transfer learning / fine-tuning with catastrophic forgetting prevention
- Pseudo-labeling for natural image training data
- Sliding-window inference on full images
- Quantitative comparison against classical edge detectors

---

## 👨‍💻 Author

**Pranjal**  
Final Year B.Tech — Computer Science (AI/ML)  
G.H. Raisoni College of Engineering, Nagpur  

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
