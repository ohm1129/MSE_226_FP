# Models Directory

This directory contains implementations of various machine learning models for online shoppers purchase prediction.

## Model Notebooks

### 1. [baselines.ipynb](baselines.ipynb)
**Purpose**: Establish baseline performance

**Models**:
- Majority Class Classifier (always predict "No Purchase")
- Logistic Regression (no tuning)

**Key Metrics**: F1-Score, ROC-AUC, Precision, Recall

---

### 2. [regularized_logistic.ipynb](regularized_logistic.ipynb)
**Purpose**: Manage variance through regularization

**Models**:
- **L1 (Lasso)**: Feature selection by driving coefficients to zero
- **L2 (Ridge)**: Shrinks coefficients but keeps all features

**Hyperparameters Tuned**:
- `C`: Inverse of regularization strength

**Key Features**:
- Feature standardization within CV to prevent leakage
- GridSearchCV for hyperparameter tuning

---

### 3. [tree_models.ipynb](tree_models.ipynb)
**Purpose**: Capture nonlinear feature interactions

**Models**:
- **Decision Tree**: Low bias, high variance
- **Random Forest**: Ensemble of trees, reduces variance through bagging

**Hyperparameters Tuned**:
- `max_depth`: Controls tree complexity
- `min_samples_split`, `min_samples_leaf`: Prevent overfitting
- `n_estimators` (RF): Number of trees
- `max_features` (RF): Random feature selection

**Key Features**:
- Feature importance analysis
- Bias-variance tradeoff exploration

---

### 4. [xgboost_model.ipynb](xgboost_model.ipynb)
**Purpose**: Balance bias reduction with variance control

**Model**: XGBoost (Gradient-Boosted Tree Ensemble)

**Key Features**:
- **Gradient Boosting**: Sequentially builds trees to correct errors
- **Shrinkage**: Learning rate controls bias-variance tradeoff
- **Regularization**: L1/L2 penalties on leaf weights
- **Built-in pruning**: Prevents overfitting

**Hyperparameters Tuned**:
- `n_estimators`: Number of boosting rounds
- `learning_rate`: Shrinkage parameter
- `max_depth`: Tree complexity
- `subsample`: Row sampling fraction
- `colsample_bytree`: Column sampling fraction
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization

**Key Features**:
- RandomizedSearchCV for efficient tuning
- Feature importance analysis
- Learning curves

---

### 5. [neural_network.ipynb](neural_network.ipynb)
**Purpose**: Test flexible nonlinear models

**Model**: Multi-Layer Perceptron (MLP)

**Architecture**:
- Input layer: 17 features
- Hidden layers: Fully connected with ReLU activation
- Output layer: Single neuron with sigmoid (binary classification)

**Regularization**:
- L2 penalty (`alpha`)
- Early stopping
- Adaptive learning rate

**Hyperparameters Tuned**:
- `hidden_layer_sizes`: Network architecture
- `alpha`: L2 regularization strength
- `learning_rate_init`: Initial learning rate

**Key Features**:
- Feature standardization (required for neural networks)
- RandomizedSearchCV for architecture search
- Loss curves visualization

---

## Shared Utilities

### [evaluate_model.py](evaluate_model.py)
Standardized model evaluation function used across all notebooks.

**Features**:
- 5-fold stratified cross-validation
- Comprehensive metrics (F1, Precision, Recall, ROC-AUC, Accuracy)
- Mean ± standard deviation reporting
- Confusion matrix
- Full training set performance

---

## Model Selection Workflow

1. **Baseline Models** → Establish minimum performance
2. **Regularized Linear Models** → Test L1/L2 regularization
3. **Tree-Based Models** → Explore nonlinear interactions
4. **XGBoost** → Advanced boosting with regularization
5. **Neural Networks** → Highly flexible nonlinear models

### Evaluation Criteria
- **Primary Metric**: F1-Score (handles class imbalance)
- **Secondary Metric**: ROC-AUC (discriminative ability)
- **Supporting Metrics**: Precision, Recall

### Bias-Variance Tradeoff
- **High Bias, Low Variance**: Logistic Regression
- **Balanced**: Regularized models, Random Forest
- **Low Bias, High Variance**: Deep Decision Trees, Complex Neural Networks
- **Optimal Balance**: XGBoost (with proper tuning)

---

## Usage

### Running a Notebook
```bash
# Navigate to models directory
cd models

# Run in Jupyter
jupyter notebook baselines.ipynb
```

### Requirements
All notebooks require:
- Preprocessed training data: `../data/train_data_preprocessed.csv`
- The `evaluate_model.py` utility in the same directory

### Consistent Methodology
All notebooks follow MS&E 226 principles:
- ✅ 5-fold stratified cross-validation
- ✅ No data leakage (preprocessing fit only on training folds)
- ✅ Performance reported as mean ± standard deviation
- ✅ Random seed set for reproducibility
- ✅ Holdout set untouched until final evaluation

---

## Next Steps

After completing all model notebooks:

1. **Compare Models**: Create comparison table across all models
2. **Select Best Model**: Based on F1-Score and ROC-AUC
3. **Final Evaluation**: Test selected model on holdout set (Part 2)
4. **Analysis**:
   - Which model best handles the bias-variance tradeoff?
   - Do nonlinear models improve over linear baselines?
   - What features are most important?
