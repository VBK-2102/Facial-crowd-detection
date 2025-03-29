# Crowd Detection using Deep Learning

## 📌 Project Overview
This project focuses on crowd detection using deep learning techniques. It utilizes OpenCV and TensorFlow/Keras to process images and detect crowded areas. The model is trained on a dataset from Kaggle to improve accuracy and efficiency in detecting crowds in various environments.

---

## 📂 Dataset
The dataset for this project can be downloaded from Kaggle.

### 🔽 How to Download the Dataset
1. Go to the dataset page on Kaggle: [Crowd Counting Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/msambare/fer2013))
2. Click on **Download**.
3. Extract the dataset and place it in the `dataset/` directory within the project folder.

Alternatively, you can use Kaggle API:
```bash
pip install kaggle  # Install Kaggle API if not already installed
mkdir dataset  # Create dataset directory
kaggle datasets download -d <dataset-name> -p dataset/ --unzip
```

---

## 🚀 Installation and Setup
Follow these steps to set up and run the project:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VBK-2102/crowd-detection.git
cd crowd-detection
```

### 2️⃣ Install Dependencies
Make sure you have Python installed (preferably version 3.8 or above).
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Training Script
To train the model, execute:
```bash
python train_emotion_model.py
```

### 4️⃣ Run the Crowd Detection Model
After training, run the following command to test the model:
```bash
python crowd_detection.py
```

---

## 🛠️ Technologies Used
- **Python**
- **OpenCV**
- **TensorFlow/Keras**
- **NumPy**
- **Matplotlib**

---

## 📢 Troubleshooting
If you encounter a NumPy compatibility issue, downgrade NumPy:
```bash
pip install numpy==1.24.3
```
If OpenCV fails to import, try reinstalling it:
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👨‍💻 Author
**Vaibhav Kalungada**  
- GitHub: [VBK-2102](https://github.com/VBK-2102)
- LinkedIn: [linkedin.com/vaibhav-kalungada-844790223/](https://linkedin.com/vaibhav-kalungada-844790223/)
