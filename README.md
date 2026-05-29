# 🏏 IPL Match Winner Prediction Model
### AI/ML Internship — Week 3 Project

---

## 📁 Project Structure
```
ipl_project/
├── data/
│   └── matches.csv          ← IPL match data (2007–2024)
├── models/                  ← Saved ML models (auto-created on run)
├── ipl_predictor.py         ← Main Python script (run this!)
├── IPL_Prediction.ipynb     ← Jupyter Notebook with explanations
├── dashboard.html           ← Interactive dashboard (open in browser)
└── requirements.txt         ← Required Python packages
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the ML script
```bash
python ipl_predictor.py
```

### 3. Open the notebook (VS Code)
- Open `IPL_Prediction.ipynb` in VS Code
- Install the Jupyter extension if prompted
- Run cells one by one with Shift+Enter

### 4. View the dashboard
- Open `dashboard.html` in any browser
- Select teams, venue, toss details
- Click "Predict Winner"!

---

## 🤖 Models Used
| Model | Type | Use Case |
|---|---|---|
| Decision Tree | Classifier | Simple, explainable |
| Random Forest | Classifier | Best accuracy (100 trees) |

## 📊 Dataset
- **Source:** Kaggle IPL Complete Dataset
- **Matches:** 1095 (2007–2024)
- **Features:** team1, team2, venue, toss_winner, toss_decision
- **Target:** winner

---
*Built for AI/ML Internship Week 3 — IPL Match Winner Prediction*
