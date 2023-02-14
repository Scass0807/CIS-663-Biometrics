import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("Assignment-1/s048r_202301.txt", sep="\t")
print(df)
df["test.out"][df["test.out"] != "s048"] = "Not s048"
df["test.subject"][df["test.subject"] != "s048"] = "Not s048"
print(df)
cm = confusion_matrix(
    df["test.subject"].to_numpy(),
    df["test.out"].to_numpy(),
    labels=["Not s048", "s048"],
)
df_cm = pd.DataFrame(cm, index=["Not s048", "s048"], columns=["Not s048", "s048"])
print(cm)
sn.heatmap(df_cm, fmt=".0f", annot=True)
# print(cm.shape)
plt.show()
