import torch
import pandas as pd
import numpy as np
import joblib
from models.informer import Informer


# === USER INPUT ===
location_num = int(input("Enter location index (0–14): "))
target_month = int(input("Enter target month (1–12): "))
target_year = int(input("Enter target year (e.g., 25 for 2025): "))
predict_bod()


