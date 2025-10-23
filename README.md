# 😷 Face Mask and Drowsiness Detection

## 📘 Abstract
After COVID-19, many changes in daily life have become mandatory for our safety — such as **wearing a face mask** and **maintaining social distance**. Ensuring the implementation of these SOPs manually is difficult, especially in public and institutional settings.  

This project proposes a **computer vision-based application** that can automatically detect:
- Whether a person is wearing a **face mask**
- Whether **social distance** is maintained
- Whether a person appears **drowsy**
- Whether a person is **distracted**

Using existing camera setups (e.g., in classrooms or offices), our system processes video frames to identify mask usage, detect drowsiness, and measure focus in real time.  

---

## 🧠 1. Introduction
**Digital Image Processing (DIP)** involves manipulating images using digital computers and algorithms to improve image quality or extract useful information.  
In our project, we process videos frame-by-frame to detect:

- Face masks  
- Drowsiness  

The first step in each module is **Face Detection**. Using **TensorFlow** and **Keras**, we trained a model to detect faces and apply further analysis on them.  

---

## ⚙️ 1.1 TensorFlow
**TensorFlow** is a machine learning library by Google used for numerical computation and deep learning applications.

### Applications
- Recognition: voice, image, and object detection  
- Motion and distance detection  
- Text-based processing (tweets, comments, etc.)

### Our Workflow
1. Environment setup  
2. Acquiring dataset of videos for testing and training  
3. Model training  
4. Model testing  

After these steps, our model can detect faces, face masks, and drowsiness.

---

## ⚙️ 1.2 Keras
**Keras** is a high-level deep learning API supporting CNNs and RNNs — ideal for image processing tasks.

### Workflow
1. Acquire and load dataset  
2. Define and compile model as a sequence of layers  
3. Execute model on data  
4. Evaluate performance  
5. Adapt model to predict new data  

---

## ⚙️ 1.3 Deep Neural Networks (DNN)
**DNNs** train models to recognize patterns similarly to how the human brain learns.

### Applications
- Image segmentation and classification  
- Object, person, and face detection  
- Depth and pose estimation  

We used the **OpenCV DNN module** for real-time object detection, as it supports TensorFlow and has fast inference time.

---

## 🧩 2. Module 1: Face Mask Detection

### 🧰 Requirements
TensorFlow >= 1.15.2
Keras == 2.3.1
Imutils == 0.5.3
Numpy == 1.18.2
opencv-python == 4.2.0.*
matplotlib == 3.2.1
scipy == 1.4.1


### 📂 Dataset
The dataset consists of:
- **1915 images with masks**
- **1918 images without masks**  
Source: [GitHub - Face Mask Detection Dataset](https://github.com/balajisrinivas/Face-Mask-Detection)

---

### 🧮 Data Preprocessing
1. Created two lists: `Data` (images) and `Labels` (mask / no mask).
2. Converted categorical labels to binary using `LabelBinarizer` from `sklearn.preprocessing`.
3. Converted lists to NumPy arrays.
4. Split dataset into train and test sets using `train_test_split` (test size = 20%).

---

### 🧠 Base Model
We used **MobileNetV2** as the base model for mask detection because it’s computationally efficient and suitable for embedded systems.

**Parameters:**
- Epochs = 20  
- Batch size = 42  
- Learning rate = 1e-4  

These settings ensure stable convergence and fast accuracy.

---

### 🏋️‍♀️ Training
Instead of using a standard CNN, we used **MobileNetV2** as the feature extractor.  
Steps:
1. Input image (array form) → Deep Neural Network  
2. Feature extraction with MobileNetV2  
3. Max pooling (7x7) → Flatten → Fully connected layer → Output  

---

### 🎥 Real-Time Detection
File: `detect_mask_video.py`  

1. The model loads pretrained weights for face and mask detection.  
2. `cv2.dnn.readNet()` is used to initialize the DNN model.  
3. `cv2.VideoStream()` captures live video.  
4. Each frame is analyzed to determine mask presence.  

---

### ✅ Result
The trained model successfully detects faces and identifies whether masks are worn in real time.

---

## 🧩 3. Module 2: Drowsiness Detection

### 🧰 Requirements

TensorFlow
Keras
OpenCV (cv2)
Numpy

---

### 📂 Dataset
Dataset contains images of **eyes (open and closed)**.  

- Training set: 1234 images  
- Validation set: 218 images  
Source: [Kaggle - Yawn & Eye Dataset](https://www.kaggle.com/serenaraju/yawn-eye-dataset-new)

---

### 🧠 Base Model
We used a **Convolutional Neural Network (CNN)** with 3 convolution layers and max pooling after each layer.

Architecture:
1. Input layer  
2. 3 Convolution + Max Pooling layers  
3. Flatten layer  
4. Fully connected layer  
5. Output layer  

Trained using Keras’ `model.fit_generator()` and saved for real-time inference.

---

### 🎥 Real-Time Detection Process

#### Step 1: Video Capture
Using OpenCV’s `cv2.VideoCapture()` and `.read()` to capture video frames.

#### Step 2: ROI Extraction
Used Haar cascade classifiers for:
- Face detection  
- Left eye detection  
- Right eye detection  

Extracted ROI (x, y, w, h) values from frames using grayscale images.

#### Step 3: Preprocessing
1. Extracted eye regions  
2. Converted to grayscale  
3. Resized using `cv2.resize()` to match dataset dimensions  
4. Normalized (divided by 255)

#### Step 4: Prediction
The pre-trained CNN model predicts whether eyes are open or closed.  
- Both eyes open → **Awake**  
- Both eyes closed → **Sleeping**  
Timer adds score for duration of eye closure.

#### Step 5: Output
Results displayed in real time using `cv2.putText()`.

---

### ✅ Result
Real-time detection accurately identifies when a person is drowsy or awake.

---

## 🚀 4. Future Trends & Applications
The applications of this system extend beyond COVID safety:

- **Public Safety:** Monitor mask compliance in schools, offices, banks, and hospitals.  
- **Smart Cities:** Automated detection through CCTV systems.  
- **Hospitals:** Alerts when staff enter wards without masks.  
- **Education:** Detect drowsy or distracted students in classrooms.  
- **Transportation:** Detect sleepy drivers to prevent road accidents.  
- **Research:** Extend using CNN-based models for higher accuracy.

---

## 📚 References
1. [What is a Deep Neural Network](https://www.oticon.com/blog/what-is-a-deep-neural-network-dnn)  
2. [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip#windows_1)  
3. [Face Mask Detection Dataset - GitHub](https://github.com/balajisrinivas/Face-Mask-Detection)  
4. [Epoch in Neural Networks - Baeldung](https://www.baeldung.com/cs/epoch-neural-networks)  
5. [Max Pooling Explanation - PapersWithCode](https://paperswithcode.com/method/max-pooling)  
6. [Yawn & Eye Dataset - Kaggle](https://www.kaggle.com/serenaraju/yawn-eye-dataset-new)

---

**Developed By:**  
👩‍💻 *Urooj Fatima*  
📅 *2021*
