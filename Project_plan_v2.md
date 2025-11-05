# 🧠 MS&E 226 Project Plan — Online Shoppers Purchasing Intention Dataset

## 📋 Executive Summary

**Objective**: Build predictive models to determine whether a website visitor will make a purchase (Revenue: 1 = yes, 0 = no)

**Dataset**: UCI Online Shoppers Purchasing Intention Dataset (12,330 sessions, 17 features)

**Stakeholder**: E-commerce marketing team seeking to improve conversion rates and ad targeting

**Class Imbalance**: ~85% no purchase, ~15% purchase

---

## 🎯 Project Structure Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PART 1: MODEL DEVELOPMENT                        │
└─────────────────────────────────────────────────────────────────────┘

📊 STAGE 1: Feature Engineering (Logistic Regression ONLY)
   ↓
   Input: train_data.csv
   ↓
   Test: Level 0 → Level 1 → Level 2 → Level 3 → Level 4
         (Basic)   (+Scale)  (+Derived) (+Inter.) (Select)
   ↓
   Output: ✅ Best Feature Set
   ↓
   ├─────────────────────────────────────────────┐
   ↓                                             ↓

🤖 STAGE 2: Model Selection (ALL models, rough params)
   ↓
   Input: Best Feature Set from Stage 1
   ↓
   Test: Logistic Reg | Random Forest | XGBoost | Neural Net
         (baseline)     (n=100, d=10)   (lr=0.1)  (64,32)
   ↓
   Compare F1-Scores
   ↓
   Output: ✅ Top 2-3 Models
   ↓
   ├─────────────────────────────────────────────┐
   ↓                                             ↓

🎯 STAGE 3: Hyperparameter Tuning (TOP 2-3 models ONLY)
   ↓
   Input: Top 2-3 Models from Stage 2
   ↓
   Tune: GridSearchCV or RandomizedSearchCV
         (Fine-tune hyperparameters)
   ↓
   Output: ✅ Single Best Model + Optimal Hyperparameters
   ↓
   ├─────────────────────────────────────────────┐
   ↓                                             ↓

┌─────────────────────────────────────────────────────────────────────┐
│                    PART 2: FINAL EVALUATION                         │
└─────────────────────────────────────────────────────────────────────┘

🎚️ Threshold Tuning
   ↓
   Input: Best Model from Stage 3
   ↓
   Optimize: Classification threshold for max F1-Score
   ↓
   Output: ✅ Optimal Threshold

🎯 Holdout Evaluation (⚠️ ONE TIME ONLY!)
   ↓
   Input: Best Model + Optimal Threshold
   ↓
   Evaluate: On holdout_data.csv (20% untouched)
   ↓
   Output: ✅ Final Performance Metrics
   ↓
   📊 DONE!
```

---

# PART 1: MODEL DEVELOPMENT & SELECTION

## ✅ 1. Data Preparation (COMPLETED)

### What We've Done:
1. ✅ **Data Loading**: `scripts/load_dataset.py` fetches raw data from UCI
2. ✅ **Train/Holdout Split**: `scripts/create_holdout_set.py` creates 80/20 stratified split
   - `data/train_data.csv` (80%) - for model development
   - `data/holdout_data.csv` (20%) - UNTOUCHED until Part 2
3. ✅ **Cross-Validation Setup**: 5-fold StratifiedKFold within training data
4. ✅ **Basic Preprocessing**: `scripts/preprocess.py` handles categorical encoding

### Data Overview:
- **Total**: 12,330 sessions
- **Train**: 9,864 sessions (80%)
- **Holdout**: 2,466 sessions (20%)
- **Target Distribution**: 85% no purchase, 15% purchase
- **Features**: 17 total (10 numeric, 7 categorical/boolean)

---

## 🔄 STAGE 1: Feature Engineering with Logistic Regression

**Strategy**: Use ONLY Logistic Regression to test feature engineering incrementally.

**Goal**: Find the best feature set that maximizes F1-Score and ROC-AUC.

**Why Logistic Regression?**
- Fast to train (test many feature combinations quickly)
- Interpretable (can see feature coefficients)
- Good baseline for understanding feature importance
- Once we find best features → use them for ALL models in Stage 2

### Pipeline Levels (Test Each Level):

#### 📊 **Level 0: CURRENT BASELINE** ✅ DONE
**What**: Basic encoding only
- Categorical encoding: LabelEncoder (Month, VisitorType)
- Boolean to int: Weekend, Revenue → 0/1
- **No scaling, no derived features**

**Status**: ✅ Implemented in `scripts/preprocess.py`
**Output**: `data/train_data_preprocessed.csv`
**Models to Run**: All baselines and advanced models

---

#### 📊 **Level 1: ADD STANDARDIZATION** 🔜 TODO
**What**: Level 0 + standardize numeric features
- One-Hot Encoding for categoricals (instead of LabelEncoder)
- StandardScaler for numeric features
- Use Pipeline to prevent leakage

**Implementation**:
```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_features = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

categorical_features = ['Month', 'VisitorType', 'Weekend']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

# Use in pipeline with model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])
```

**Why**:
- Neural networks require standardized features
- Some models (Logistic Regression, SVM) benefit from scaling
- OneHot encoding better than LabelEncoder for non-ordinal categoricals

**Compare**: F1-Score, ROC-AUC vs Level 0

---

#### 📊 **Level 2: ADD DERIVED FEATURES** 🔜 TODO
**What**: Level 1 + engineered behavioral features

**New Features to Create**:
```python
def add_derived_features(X):
    X = X.copy()

    # Average time per page visited
    X["avg_time_per_page"] = (
        X["ProductRelated_Duration"] + X["Informational_Duration"] + X["Administrative_Duration"]
    ) / (X["ProductRelated"] + X["Informational"] + X["Administrative"] + 1e-5)

    # Total session duration
    X["total_duration"] = (
        X["ProductRelated_Duration"] +
        X["Informational_Duration"] +
        X["Administrative_Duration"]
    )

    # Bounce to exit ratio
    X["bounce_exit_ratio"] = X["BounceRates"] / (X["ExitRates"] + 1e-5)

    # Interaction intensity
    X["interaction_intensity"] = (
        X["ProductRelated"] + X["Informational"] + X["Administrative"]
    )

    return X
```

**Pipeline**:
```python
from sklearn.preprocessing import FunctionTransformer

feature_creator = FunctionTransformer(add_derived_features)

pipeline = Pipeline([
    ("feature_creator", feature_creator),
    ("preprocessor", preprocessor),
    ("classifier", model)
])
```

**Compare**: F1-Score, ROC-AUC vs Level 1

---

#### 📊 **Level 3: ADD INTERACTION TERMS** 🔜 TODO (Optional)
**What**: Level 2 + interaction features

**Potential Interactions**:
- `VisitorType × Month` (returning visitors in December?)
- `VisitorType × Weekend`
- `PageValues × ProductRelated_Duration`

**Implementation**:
```python
from sklearn.preprocessing import PolynomialFeatures

# Create interactions for specific features
interaction_features = ['VisitorType', 'Month', 'Weekend', 'PageValues']
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
```

**Compare**: F1-Score, ROC-AUC vs Level 2

---

#### 📊 **Level 4: FEATURE SELECTION** 🔜 TODO
**What**: Keep only most important features

**Methods to Try**:
1. **Correlation Analysis**: Remove highly correlated features (r > 0.9)
2. **Tree-based Importance**: Use Random Forest/XGBoost feature importances
3. **L1 Regularization**: Logistic Regression with L1 penalty
4. **Recursive Feature Elimination (RFE)**

**Goal**: Reduce overfitting, improve generalization

**Compare**: F1-Score, ROC-AUC vs Level 3

---

---

## 🤖 STAGE 2: Model Selection (Rough Parameters)

**Input**: Best feature set from Stage 1

**Strategy**: Test ALL models with default or rough hyperparameters

**Goal**: Identify top 2-3 models that perform best with the selected features

### Models to Test (with rough/default parameters):

#### 1. **Baseline: Logistic Regression**
Already tested in Stage 1 - use best result as baseline

#### 2. **Random Forest**
Use default parameters or rough settings:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

#### 3. **XGBoost**
Use default parameters or rough settings:
```python
xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

#### 4. **Neural Network**
Use simple architecture:
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    alpha=0.001,
    max_iter=200,
    early_stopping=True,
    random_state=42
)
```

### Evaluation:
- **Cross-Validation**: 5-Fold StratifiedKFold
- **Metrics**: F1-Score (primary), ROC-AUC, Precision, Recall
- **No hyperparameter tuning yet** - just test with rough params

### Output:
**Ranking of models** by F1-Score → Select top 2-3 for Stage 3

---

## 🎯 STAGE 3: Hyperparameter Tuning (Top Models Only)

**Input**: Top 2-3 models from Stage 2

**Strategy**: Fine-tune ONLY the best performing models

**Goal**: Find optimal hyperparameters for final model selection

### Hyperparameter Grids for Tuning:

#### If Random Forest is in Top 3:
```python
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
# Use RandomizedSearchCV with n_iter=30
```

#### If XGBoost is in Top 3:
```python
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1.0, 10.0]
}
# Use RandomizedSearchCV with n_iter=50
```

#### If Neural Network is in Top 3:
```python
param_grid_nn = {
    'classifier__hidden_layer_sizes': [
        (32,), (64,), (128,),
        (64, 32), (128, 64), (128, 64, 32)
    ],
    'classifier__alpha': [0.0001, 0.001, 0.01],
    'classifier__learning_rate_init': [0.001, 0.01]
}
# Use RandomizedSearchCV with n_iter=20
```

### Evaluation:
- **Cross-Validation**: 5-Fold StratifiedKFold
- **Search Method**: GridSearchCV (small grids) or RandomizedSearchCV (large grids)
- **Scoring**: F1-Score (primary metric)
- **Report**: Mean ± std for F1, ROC-AUC, Precision, Recall

### Output:
**Single best model** with optimal hyperparameters → Ready for Part 2

---

### 🎯 Model Comparison Strategy

Create a comparison table like this:

| Feature Level | Model | F1-Score | ROC-AUC | Precision | Recall | Notes |
|--------------|-------|----------|---------|-----------|--------|-------|
| Level 0 (Basic) | Baseline LR | 0.48 ± 0.02 | 0.86 ± 0.01 | - | - | Simple baseline |
| Level 0 | L1 Logistic | 0.50 ± 0.02 | 0.87 ± 0.01 | - | - | Best C=0.1 |
| Level 0 | Random Forest | 0.52 ± 0.03 | 0.88 ± 0.01 | - | - | 300 trees |
| Level 0 | XGBoost | 0.54 ± 0.02 | 0.89 ± 0.01 | - | - | Best so far |
| Level 0 | Neural Net | 0.51 ± 0.03 | 0.87 ± 0.02 | - | - | (64,32) layers |
| Level 1 (+ Scaling) | XGBoost | 0.55 ± 0.02 | 0.90 ± 0.01 | - | - | Improved! |
| Level 2 (+ Derived) | XGBoost | 0.56 ± 0.02 | 0.91 ± 0.01 | - | - | Best features |
| ... | ... | ... | ... | ... | ... | ... |

**Goal**: Identify best (Model × Feature Level) combination

---

## 📈 4. Model Selection (End of Part 1)

### Selection Criteria:

1. **Primary**: Highest mean F1-Score
2. **Secondary**: Highest ROC-AUC
3. **Considerations**:
   - Stability (low std across folds)
   - Computational cost
   - Interpretability (if stakeholder cares)
   - Overfitting risk (train vs. validation gap)

### Deliverables:

- ✅ Comparison table of all models
- ✅ Selected best model + hyperparameters
- ✅ Feature engineering level used
- ✅ Justification for selection

---

# PART 2: FINAL EVALUATION

## 🎚️ 5. Threshold Tuning

**Input**: Best model from Part 1

**Goal**: Optimize classification threshold for F1-Score

### Process:

1. **Generate predictions** on validation set (from 5-fold CV):
   ```python
   y_proba = best_model.predict_proba(X_val)[:, 1]
   ```

2. **Sweep thresholds** from 0.1 to 0.9 (step=0.05):
   ```python
   thresholds = np.arange(0.1, 0.9, 0.05)
   for threshold in thresholds:
       y_pred = (y_proba >= threshold).astype(int)
       f1 = f1_score(y_val, y_pred)
       precision = precision_score(y_val, y_pred)
       recall = recall_score(y_val, y_pred)
   ```

3. **Select optimal threshold**: Max F1-Score

4. **Visualize**:
   - Precision-Recall curve
   - F1-Score vs. Threshold plot

5. **Save optimal threshold** for holdout evaluation

---

## 🎯 6. Holdout Set Evaluation (FINAL)

**⚠️ WARNING**: Run this ONLY ONCE at the very end!

### Process:

1. **Preprocess holdout set**:
   ```bash
   python scripts/preprocess.py data/holdout_data.csv
   ```

2. **Load best model** (trained on full training set)

3. **Apply optimal threshold**:
   ```python
   y_holdout_proba = best_model.predict_proba(X_holdout)[:, 1]
   y_holdout_pred = (y_holdout_proba >= optimal_threshold).astype(int)
   ```

4. **Compute final metrics**:
   - F1-Score
   - ROC-AUC
   - Precision
   - Recall
   - Confusion Matrix
   - Classification Report

5. **Analyze**:
   - Compare holdout metrics to CV estimates
   - Identify any overfitting
   - Feature importance analysis
   - Error analysis (false positives vs. false negatives)

---

## 📋 7. Final Deliverables

### Part 1 Deliverables:
- ✅ Comparison table: all models × feature levels
- ✅ Selected best model + hyperparameters
- ✅ Cross-validation results (mean ± std)
- ✅ Feature engineering analysis
- ✅ Model selection justification

### Part 2 Deliverables:
- ⬜ Threshold tuning analysis
- ⬜ Holdout set evaluation (FINAL metrics)
- ⬜ Comparison: CV vs. Holdout performance
- ⬜ Feature importance analysis
- ⬜ Error analysis
- ⬜ Recommendations for stakeholder

---

## 🔄 Current Status & Next Steps

### ✅ What's Done:
1. ✅ Data loading pipeline (`scripts/load_dataset.py`)
2. ✅ Train/holdout split (80/20) (`scripts/create_holdout_set.py`)
3. ✅ Basic preprocessing Level 0 (`scripts/preprocess.py`)
4. ✅ Evaluation utility (`models/evaluate_model.py`)
5. ✅ Model notebooks created:
   - `models/baselines.ipynb`
   - `models/regularized_logistic.ipynb`
   - `models/tree_models.ipynb`
   - `models/xgboost_model.ipynb`
   - `models/neural_network.ipynb`

---

### 🔜 STAGE 1: Feature Engineering (Logistic Regression Only)

**Goal**: Find best feature set using Logistic Regression

**Tasks**:
1. ⬜ **Level 0** (Done): Run Logistic Regression with current preprocessing
   - Already have: `data/train_data_preprocessed.csv`
   - Record: F1-Score, ROC-AUC

2. ⬜ **Level 1**: Add Standardization + OneHot Encoding
   - Create new preprocessing pipeline with StandardScaler
   - Run Logistic Regression
   - Compare: Level 1 vs Level 0

3. ⬜ **Level 2**: Add Derived Features
   - Add: avg_time_per_page, total_duration, bounce_exit_ratio, interaction_intensity
   - Run Logistic Regression
   - Compare: Level 2 vs Level 1

4. ⬜ **Level 3** (Optional): Add Interaction Terms
   - Add: VisitorType × Month, VisitorType × Weekend, etc.
   - Run Logistic Regression
   - Compare: Level 3 vs Level 2

5. ⬜ **Level 4**: Feature Selection
   - Method 1: Correlation analysis (remove r > 0.9)
   - Method 2: L1 Logistic Regression (sparse features)
   - Method 3: Feature importances from tree model
   - Run Logistic Regression with selected features
   - **Final Output**: Best feature set

**Deliverable**: Optimal feature set to use in Stage 2

---

### 🔜 STAGE 2: Model Selection (Rough Parameters)

**Goal**: Test all models with rough params on best feature set

**Tasks**:
1. ⬜ Update preprocessing to use best feature set from Stage 1
2. ⬜ Run Random Forest (default/rough params)
3. ⬜ Run XGBoost (default/rough params)
4. ⬜ Run Neural Network (simple architecture)
5. ⬜ Create comparison table:

| Model | F1-Score | ROC-AUC | Precision | Recall | Notes |
|-------|----------|---------|-----------|--------|-------|
| Logistic Regression | X.XX ± 0.XX | X.XX ± 0.XX | ... | ... | From Stage 1 |
| Random Forest | X.XX ± 0.XX | X.XX ± 0.XX | ... | ... | n_est=100, depth=10 |
| XGBoost | X.XX ± 0.XX | X.XX ± 0.XX | ... | ... | lr=0.1, depth=6 |
| Neural Net | X.XX ± 0.XX | X.XX ± 0.XX | ... | ... | (64,32) layers |

6. ⬜ Select top 2-3 models based on F1-Score

**Deliverable**: Top 2-3 candidate models

---

### 🔜 STAGE 3: Hyperparameter Tuning (Top Models Only)

**Goal**: Fine-tune only the best 2-3 models

**Tasks**:
1. ⬜ Set up hyperparameter grids for top models
2. ⬜ Run GridSearchCV or RandomizedSearchCV
3. ⬜ Compare tuned models
4. ⬜ Select single best model
5. ⬜ Train final model on full training set

**Deliverable**: Single best model with optimal hyperparameters

---

## 📊 Expected Timeline

| Stage | Tasks | Estimated Time |
|-------|-------|----------------|
| **STAGE 1** | Feature Engineering (Logistic Regression) |  |
| → Level 0 | Current preprocessing | ✅ Done |
| → Level 1 | + Standardization | 30 min |
| → Level 2 | + Derived features | 30 min |
| → Level 3 | + Interactions (optional) | 30 min |
| → Level 4 | Feature selection | 1 hour |
| | **Stage 1 Subtotal** | **~2.5 hours** |
| **STAGE 2** | Model Selection (Rough Params) |  |
| → Setup | Preprocess with best features | 15 min |
| → Models | Run 4 models (RF, XGB, NN + LR baseline) | 1-2 hours |
| → Analysis | Compare and select top 2-3 | 30 min |
| | **Stage 2 Subtotal** | **~2-3 hours** |
| **STAGE 3** | Hyperparameter Tuning (Top Models) |  |
| → Tuning | GridSearch/RandomizedSearch on 2-3 models | 2-4 hours |
| → Selection | Select final best model | 30 min |
| | **Stage 3 Subtotal** | **~2.5-4.5 hours** |
| **PART 2** | Final Evaluation |  |
| → Threshold | Optimize classification threshold | 1 hour |
| → Holdout | Final evaluation (ONE TIME) | 30 min |
| → Report | Analysis & write-up | 2 hours |
| | **Part 2 Subtotal** | **~3.5 hours** |
| | **GRAND TOTAL** | **~10-13 hours** |

---

## 🎯 Success Criteria

### Minimum Targets:
- F1-Score > 0.50 (beat baseline)
- ROC-AUC > 0.87 (beat baseline)

### Good Performance:
- F1-Score > 0.55
- ROC-AUC > 0.90

### Excellent Performance:
- F1-Score > 0.60
- ROC-AUC > 0.92

### Additional Goals:
- ✅ No data leakage
- ✅ Proper cross-validation
- ✅ Clear comparison across models
- ✅ Reproducible results (random seeds)
- ✅ Well-documented process

---

## 🛠️ Technical Notes

### Leakage Prevention Checklist:
- ✅ Holdout set created BEFORE any exploration
- ✅ Preprocessing fit only on training folds (via Pipeline)
- ✅ No peeking at holdout set until final evaluation
- ✅ Cross-validation with StratifiedKFold
- ✅ Random seeds set for reproducibility

### Computational Considerations:
- GridSearchCV can be slow for large grids
- Use RandomizedSearchCV for large parameter spaces
- Limit to ~30-50 configurations per model
- Run models in parallel when possible
- Save trained models for later use

### Key Files:
```
MSE_226_FP/
├── data/
│   ├── full_online_shoppers_data.csv (raw)
│   ├── train_data.csv (raw train split)
│   ├── train_data_preprocessed.csv (Level 0)
│   └── holdout_data.csv (UNTOUCHED)
├── scripts/
│   ├── load_dataset.py
│   ├── create_holdout_set.py
│   └── preprocess.py
├── models/
│   ├── evaluate_model.py (shared utility)
│   ├── baselines.ipynb ✅
│   ├── regularized_logistic.ipynb
│   ├── tree_models.ipynb
│   ├── xgboost_model.ipynb
│   └── neural_network.ipynb
└── Project_plan_v2.md (this file)
```

---

---

## 🎯 QUICK START GUIDE

### **Right Now - Start with Stage 1:**

1. **Create a new notebook**: `models/feature_engineering.ipynb`

2. **Test these feature levels with Logistic Regression**:
   - Level 0: Current preprocessing (already done) ✅
   - Level 1: + StandardScaler + OneHotEncoder
   - Level 2: + Derived features
   - Level 3: + Interactions (optional)
   - Level 4: Feature selection

3. **Track results**:
   ```
   | Level | Features | F1-Score | ROC-AUC |
   |-------|----------|----------|---------|
   | 0     | Basic    | X.XX     | X.XX    |
   | 1     | +Scale   | X.XX     | X.XX    |
   | 2     | +Derived | X.XX     | X.XX    |
   | ...   | ...      | ...      | ...     |
   ```

4. **Select best feature set** → Use in Stage 2

---

## 📝 THREE-STAGE SUMMARY

### **STAGE 1: Feature Engineering** (Logistic Regression only)
- **Input**: Raw training data
- **Process**: Test feature levels incrementally
- **Output**: Best feature set
- **Time**: ~2.5 hours

### **STAGE 2: Model Selection** (All models, rough params)
- **Input**: Best feature set from Stage 1
- **Process**: Test 4 models with default parameters
- **Output**: Top 2-3 models
- **Time**: ~2-3 hours

### **STAGE 3: Hyperparameter Tuning** (Top models only)
- **Input**: Top 2-3 models from Stage 2
- **Process**: Fine-tune hyperparameters
- **Output**: Single best model
- **Time**: ~2.5-4.5 hours

### **PART 2: Final Evaluation** (Holdout set)
- **Input**: Best tuned model from Stage 3
- **Process**: Threshold tuning + holdout evaluation
- **Output**: Final performance metrics
- **Time**: ~3.5 hours

---

**Total Time: ~10-13 hours to complete project**

**Let's build the best model! 🚀**
