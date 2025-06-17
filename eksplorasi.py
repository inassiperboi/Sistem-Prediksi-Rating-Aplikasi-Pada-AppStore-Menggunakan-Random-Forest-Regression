import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("dataset.csv")

# Info umum
print("=== Info Dataset ===")
print(df.info())

# Statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Cek missing value
print("\n=== Missing Values ===")
print(df.isnull().sum())

# Distribusi rating
plt.figure(figsize=(8, 5))
sns.histplot(df["Average_User_Rating"], bins=10, kde=True)
plt.title("Distribusi Average User Rating")
plt.xlabel("Rating")
plt.ylabel("Jumlah Aplikasi")
plt.show()

# Korelasi antar fitur numerik
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Heatmap Korelasi")
plt.show()

# Boxplot rating berdasarkan genre
plt.figure(figsize=(12, 6))
top_genres = df["Primary_Genre"].value_counts().nlargest(10).index
sns.boxplot(data=df[df["Primary_Genre"].isin(top_genres)], 
            x="Primary_Genre", y="Average_User_Rating")
plt.xticks(rotation=45)
plt.title("Rating Berdasarkan Genre Terpopuler")
plt.show()
