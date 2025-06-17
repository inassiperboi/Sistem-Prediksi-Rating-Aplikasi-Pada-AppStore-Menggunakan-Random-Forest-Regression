import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("dataset.csv")

# Buang baris dengan rating = 0 karena dianggap belum mendapatkan rating
df_clean = df[df["Average_User_Rating"] > 0].copy()

# Drop kolom yang tidak relevan
drop_cols = ["App_Name", "Developer", "Released", "Updated", "Version", "Currency", "Required_IOS_Version"]
df_clean = df_clean.drop(columns=drop_cols)

# Pisahkan fitur dan target
X = df_clean.drop(columns=["Average_User_Rating"])
y = df_clean["Average_User_Rating"]

# Identifikasi fitur numerik dan kategorik
num_features = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Pipeline numerik dan kategorik
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Gabungkan keduanya
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ])

# Terapkan transformasi
X_processed = preprocessor.fit_transform(X)

# Ambil nama-nama fitur hasil one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_features)
all_feature_names = num_features + list(ohe_feature_names)

# Buat dataframe hasil preprocessing
df_processed = pd.DataFrame(X_processed, columns=all_feature_names)
df_processed["Average_User_Rating"] = y.values

# Simpan ke CSV
df_processed.to_csv("dataset_preprocessed.csv", index=False)

print("✅ Data preprocessing selesai dan disimpan sebagai 'dataset_preprocessed.csv'")
