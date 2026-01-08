# Pattern Recognition System using ART Neural Network
**Adaptive Resonance Theory (ART) with Image Processing and Graphical Interface**

This project implements a **Pattern Recognition System based on the ART (Adaptive Resonance Theory) neural model** using **Python**.  
The application allows users to load images or draw patterns manually and train the system to recognize them using similarity matching.

The system supports incremental learning and stores trained patterns locally using JSON files.

---

## Features

-  Graphical user interface built with Tkinter.
-  Image loading and drawing canvas.
-  Pattern preprocessing (grayscale, thresholding, resizing).
-  ART neural network implementation from scratch.
-  Incremental training with new patterns.
-  Adjustable vigilance parameter (ρ).
-  Pattern similarity visualization.
-  Persistent pattern storage in JSON format.
-  Real-time recognition feedback.

---

## How the ART Model Works

The Adaptive Resonance Theory (ART) model is a type of neural network designed for stable and fast learning of recognition categories.

### Key Concepts

- **Vigilance parameter (ρ)**  
  Controls how similar a new pattern must be to match an existing category.

- **Similarity measurement**  
  Uses cosine similarity between pattern vectors.

- **Learning rule**  
  Weights are updated using a learning factor when similarity exceeds the vigilance threshold.

- **Incremental learning**  
  New patterns can be added without retraining the entire system.

---

## Image Preprocessing Pipeline

1. Load image or canvas drawing.
2. Convert to grayscale.
3. Apply binary thresholding.
4. Resize to 100×100 pixels.
5. Flatten into a feature vector.
6. Compare with stored patterns.

---

## Technologies Used

- Python 3
- Tkinter (GUI)
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- JSON file storage

---

By: AlanGM16
