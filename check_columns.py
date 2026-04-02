import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download('thedevastator/childhood-allergies-prevalence-diagnosis-and-tre')
filepath = os.path.join(path, 'food-allergy-analysis-Zenodo.csv')
df = pd.read_csv(filepath, nrows=5)
print("Columns in the dataset:")
for col in df.columns:
    print(col)
