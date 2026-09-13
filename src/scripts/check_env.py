import torch
import numpy as np
import pandas as pd
import cv2
import sklearn

print("=== wafer Defect Vision Environment")
print("PyTorch :", torch.__version__)
print("NumPy   :", np.__version__)
print("Pandas  :", pd.__version__)
print("OpenCV  :", cv2.__version__)
print("GPU     :", torch.cuda.is_available())

print("\nEnvironment setup complete!")