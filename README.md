# 🏏 IPL Match Winner Prediction Model

<div align="center">

![IPL Banner](https://img.shields.io/badge/IPL-Match%20Prediction-orange?style=for-the-badge&logo=cricket&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

### 🤖 Predicting IPL match winners using Machine Learning
### Built as part of AI/ML Internship — Week 3 Hands-on Project

</div>

---

## 📌 Table of Contents
- [About the Project](#about-the-project)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Screenshots](#screenshots)
- [What I Learned](#what-i-learned)

---

## 🎯 About the Project

This project builds a **Machine Learning model** that predicts which team will win an IPL match — just like cricket experts do, but using **mathematical patterns from 1095 real IPL matches (2007–2024)**.

Given match details like:
- 🏏 Team 1 vs Team 2
- 🏟️ Venue / Stadium
- 🪙 Toss winner & decision

The model predicts → **🏆 Which team wins!**

---

## 🌐 Live Demo

> Open `dashboard.html` in any browser for the interactive prediction UI!

Features of the dashboard:
- ✅ Select any two IPL teams
- ✅ Choose venue and toss details
- ✅ Get instant winner prediction with confidence %
- ✅ View all-time win rates, toss analysis, feature importance

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Kaggle — IPL Complete Dataset |
| **Matches** | 1095 IPL matches |
| **Seasons** | 2007/08 to 2024 |
| **Features Used** | team1, team2, venue, toss_winner, toss_decision |
| **Target** | winner |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data loading & manipulation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical operations |
| ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | ML models & evaluation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Data visualization |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat) | Statistical charts |
| ![HTML/CSS/JS](https://img.shields.io/badge/HTML%2FCSS%2FJS-E34F26?style=flat&logo=html5&logoColor=white) | Interactive dashboard |

---

## 📁 Project Structure

```
IPL-Match-Winner-Prediction/
│
├── 📂 data/
│   └── matches.csv              ← IPL match data (2007–2024)
│
├── 📂 models/                   ← Saved ML models (auto-created)
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   └── label_encoders.pkl
│
├── 🐍 ipl_predictor.py          ← Main ML script with deep comments
├── 📓 IPL_Prediction.ipynb      ← Jupyter Notebook with visualizations
├── 🌐 dashboard.html            ← Interactive prediction dashboard
├── 📋 requirements.txt          ← Python dependencies
└── 📖 README.md                 ← You are here!
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.x installed
- VS Code (recommended)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/IPL-Match-Winner-Prediction.git
cd IPL-Match-Winner-Prediction
```

### 2️⃣ Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3️⃣ Run the ML script
```bash
python ipl_predictor.py
```

### 4️⃣ Open the Dashboard
Simply double-click `dashboard.html` — opens in your browser! 🌐

### 5️⃣ Open the Notebook (VS Code)
- Install the Jupyter extension in VS Code
- Open `IPL_Prediction.ipynb`
- Click **Run All**

---

## 📈 Model Performance

| Model | Accuracy |
|---|---|
| 🌳 Decision Tree | **56.42%** |
| 🌲 Random Forest | 53.67% |

> **Why ~56%?** IPL cricket is inherently unpredictable — upsets happen all the time! A 56% accuracy means the model beats random guessing (50%) and captures real patterns in the data. With only pre-match features (no player form, pitch report, weather), this is a solid result.

### Classification Report Highlights
- **Chennai Super Kings** → 77% precision (most predictable team!)
- **Mumbai Indians** → 69% recall
- **Toss decision** → least important feature (only 5%)

---

## 🔑 Key Insights

```
🏟️  Venue          ████████████████  28.3%  ← Most important!
🏏  Team 1         ██████████████    24.9%
🏏  Team 2         █████████████     22.9%
🪙  Toss Winner    ██████████        18.8%
🎯  Toss Decision  ███               5.3%   ← Least important
```

### 💡 What the model discovered:
1. **Home ground advantage is REAL** — venue is the #1 predictor
2. **Team strength matters more than toss** — consistent teams win regardless
3. **Toss decision (bat/field) barely matters** — only 5% importance
4. **CSK is the most consistent team** — highest precision in predictions

---

## 🖥️ Screenshots

> Dashboard with live match prediction UI
> Confusion matrix heatmap
> Feature importance chart
> Jupyter notebook with all visualizations

---

## 🎓 What I Learned

Through this project I learned:

- ✅ **Data Preprocessing** — handling missing values, encoding categorical data
- ✅ **Label Encoding** — converting text (team names) into numbers for ML
- ✅ **Train/Test Split** — why we never test on training data (80/20 rule)
- ✅ **Decision Tree** — how yes/no questions lead to predictions
- ✅ **Random Forest** — why 100 trees beat 1 tree (ensemble learning)
- ✅ **Model Evaluation** — accuracy, confusion matrix, classification report
- ✅ **Feature Importance** — understanding WHAT drives predictions
- ✅ **Saving Models** — using joblib to persist trained models

---

## 🙋 Author

**Pallavi** — AI/ML Internship Batch (VIP)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/yourusername)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If you found this helpful, please star the repository! ⭐**

*Built with ❤️ and lots of cricket knowledge 🏏*

</div>
