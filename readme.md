# 📰 News Recommendation System on MIND Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![MIND](https://img.shields.io/badge/Dataset-MIND-green)

## 📌 Introduction

This project implements a **Deep Learning-based Recommender System** for news recommendation using the **MIND (Microsoft News Dataset)**. The goal is to predict the probability that a user will click on a candidate news article based on their historical reading behavior.

The system utilizes a **Two-Tower Architecture** (Dual-Encoder):
1.  **News Encoder:** Uses **1D-CNN** and **Attention Mechanism** to extract semantic features from news titles.
2.  **User Encoder:** Aggregates reading history using **Attention** to form user interest vectors.

The model ranks candidate news based on the dot product similarity between the user vector and candidate news vectors.

---

## 👨‍💻 Team Members

**Hanoi University of Science and Technology (HUST)**
*School of Information and Communication Technology*

| Student Name | Student ID |
|--------------|------------|
| **Nguyễn Nhật Quang** | 20224892 |
| **Vũ Đức Tài** | 20225082 |

* **Instructor:** TS. Ngô Văn Linh
* **Course:** Recommender Systems
* **Semester:** 2025.2

---

## 🧠 Model Architecture

The model follows a hierarchical structure:

* **Input:** News Titles (Text sequence) and User History (Sequence of clicked news).
* **Embedding Layer:** Converts words into 300-dimensional dense vectors.
* **News Encoder:**
    * `Conv1d`: Captures local context and phrases from titles.
    * `Attention`: Weighs important words to create a news vector.
* **User Encoder:**
    * Takes the sequence of news vectors from the user's history.
    * `Attention`: Weighs important news in the history to form the User Vector.
* **Scoring:** Computes the Dot Product between User Vector and Candidate News Vector.

---

## 📂 Project Structure

```bash
├── checkpoints/          # Saved model weights (e.g., mind_model_ep3.pth)
├── MIND_large_train/     # Training data (behaviors.tsv, news.tsv)
├── MIND_small_train/     # Small training data (used for building vocab)
├── MINDlarge_test/       # Testing data (behaviors.tsv, news.tsv)
├── model.py              # Neural Network classes (NewsEncoder, UserEncoder)
├── preprocess.py         # Data loading and text processing logic
├── train.py              # Training script
├── predict.py            # Inference script for generating submission
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
