# sift-image-matcher
A Python project for detecting and comparing images using SIFT keypoints and Bag-of-Words model, developed for the CSCI935 Computer Vision course at UOW.
# SIFT-ImageMatcher

A Python project using OpenCV 4.10.0 to detect and visualize SIFT keypoints, and to compare multiple images using a Bag-of-Words model based on SIFT descriptors.

---

## 📌 Features

### ✅ Task 1: Keypoint Detection & Visualization
- Rescales a single image to VGA size (keeping aspect ratio)
- Extracts SIFT keypoints from the luminance (Y) component
- Draws:
  - A cross at each keypoint
  - A scale-proportional circle
  - Orientation line from cross to circle
- Displays the original and annotated image side-by-side
- Prints total keypoint count to terminal

### ✅ Task 2: Image Comparison with Bag-of-Words
- Accepts multiple images
- Extracts SIFT descriptors from luminance
- Performs K-Means clustering of descriptors into visual words
- Builds histograms per image
- Computes and prints dissimilarity matrix (χ² distance)
- Supports K = 5%, 10%, 20% of total keypoints

---

## 🛠 Requirements

- Python 3.x
- OpenCV 4.10.0
- NumPy



