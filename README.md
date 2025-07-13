# 🏠 House Price Prediction using XGBoost

This project predicts house prices using the Boston Housing Dataset and the XGBoost Regressor model.

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## 🚀 Steps
1. Data loaded and analyzed with correlation heatmap.
2. Model trained using XGBoost Regressor.
3. Evaluation using R² Score and MAE.

## 📊 Model Accuracy
- R² Score (Train): 0.999
- MAE (Train): ~0.14
- R² Score (Test): ~0.87
- MAE (Test): ~2.3

## 📂 Dataset
Boston Housing Dataset (CSV)

## 🔍 How to Run
```b
pip install -r requirements.txt
python house_price_predictor.py
```

---

## ✅ Step 5: Run in PyCharm

1. Open PyCharm → `Open` → Choose the `house-price-prediction/` folder.
2. Place your `.csv` file inside the folder.
3. Install packages:
   - Open terminal in PyCharm:
     ```bash
     pip install -r requirements.txt
     ```
4. Right-click on `house_price_predictor.py` → **Run**.

📌 If graph doesn’t show: Make sure you include `plt.show()` — you already did ✔️

---

## ✅ Step 6: Upload to GitHub

1. Go to [https://github.com](https://github.com) → Create new repo → Name: `house-price-prediction`
2. In PyCharm terminal:

```bash
git init
git add .
git commit -m "Initial commit - House Price Prediction"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git
git push -u origin main
```