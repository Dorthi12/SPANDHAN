# SPANDHAN System Architecture

SPANDHAN combines Digital Signal Processing (DSP) in MATLAB with Machine Learning (ML) in Python for signal feature extraction and classification.

```
+-------------------+      +-------------------+      +--------------------+
|  Input Signal     | ---> |  Preprocessing &  | ---> | Feature Extraction |
|  (Audio / Image)  |      |  DSP Filtering    |      | (Time/Freq/Spatial)|
+-------------------+      +-------------------+      +--------------------+
                                                                |
                                                                v
+-------------------+      +-------------------+      +--------------------+
| Output Predictions| <--- | Classification &  | <--- | ML Inference Model |
| & Visualizations  |      | Parsing Results   |      | (Audio / Image)    |
+-------------------+      +-------------------+      +--------------------+
```
