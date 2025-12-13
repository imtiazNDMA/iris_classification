# 🌸 Iris Classification Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project demonstrating end-to-end classification using multiple algorithms with hyperparameter optimization and professional data visualizations.

## 🎯 Project Overview

This project showcases a complete machine learning pipeline for the classic Iris classification problem, featuring:

- **🔍 Exploratory Data Analysis** with comprehensive visualizations
- **⚙️ Data Preprocessing** with feature scaling and train-test splitting
- **🤖 Model Training** with 4 different classification algorithms
- **🎛️ Hyperparameter Optimization** using GridSearchCV
- **📊 Performance Comparison** with detailed metrics and analysis
- **📈 Professional Visualizations** with publication-ready design

## 📊 Dataset

- **Dataset**: Iris Flower Classification
- **Features**: 4 numerical measurements (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 iris species (Setosa, Versicolor, Virginica)
- **Samples**: 150 observations
- **Balance**: Perfectly balanced (50 samples per class)

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML Framework**: Scikit-learn
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Package Management**: UV
- **Development**: Jupyter Notebooks for exploration, Python scripts for production

## Models Trained

### Baseline Models
1. **Logistic Regression** - Linear classifier with L2 regularization
2. **Decision Tree** - Non-linear classifier with Gini impurity
3. **Random Forest** - Ensemble method with 100 trees
4. **Support Vector Machine** - RBF kernel with C=1.0

### Hyperparameter Tuning
- **GridSearchCV** with 5-fold cross-validation
- **Parameter Grids**: 20+ combinations per model
- **Scoring Metric**: Accuracy
- **Validation**: Stratified sampling to maintain class balance

## Results & Performance

### Final Model Rankings

| Rank | Model | Test Accuracy | CV Score | Improvement |
|------|-------|---------------|----------|-------------|
| 🥇 | **Decision Tree (Tuned)** | **97.78%** | 95.24% | +7.32% |
| 🥈 | **Support Vector Machine** | **93.33%** | 98.10% | +0.00% |
| 🥉 | **Logistic Regression** | **91.11%** | 98.10% | +0.00% |
| 4️⃣ | **Random Forest (Tuned)** | **91.11%** | 96.19% | +2.50% |

![Performance Comparison](results/ml_pipeline_detailed_metrics.png)

### Key Findings

- **Best Performer**: Decision Tree with max_depth=3 achieved 97.78% accuracy
- **Perfect Separation**: All models perfectly classified Iris-setosa (100% precision)
- **Stable Performance**: Low variance across cross-validation folds (std < 0.04)
- **Hyperparameter Impact**: Decision Tree showed highest improvement from tuning (7.32%)
- **Linear Models**: SVM and Logistic Regression were already near-optimal

## 📊 Visualizations

### Exploratory Data Analysis (`results/EDA/`)
- **Feature Distributions**: Histograms showing data distribution patterns
- **Correlation Heatmap**: Feature correlation matrix with annotations
- **Pairwise Relationships**: Scatter plot matrix with class separation
- **Box Plots**: Statistical summary and outlier detection

### Model Performance Analysis (`results/ML/`)
- **Accuracy Comparison**: Bar chart comparing model performance
- **Confusion Matrices**: Individual confusion matrices for each model
- **Cross-Validation Scores**: CV performance across different models
- **Average Metrics**: Precision, recall, and F1-score comparison
- **Improvement Analysis**: Before/after hyperparameter tuning results

## 📁 Project Structure

```
iris_classification/
├── data/                     # Dataset files
│   ├── features.csv         # Input features (150x4)
│   └── targets.csv          # Target labels (150x1)
├── notebooks/               # Jupyter notebooks for exploration
│   ├── data_ingestion.ipynb # Data loading and preparation
│   ├── EDA.ipynb           # Exploratory data analysis
│   ├── ml_pipeline.ipynb   # Machine learning pipeline
│   └── ml_vis.ipynb        # Visualization creation
├── src/                     # Production-ready Python modules
│   ├── eda_analysis.py     # Exploratory data analysis
│   ├── ml_pipeline.py      # Machine learning pipeline
│   └── ml_visualizations.py # Visualization generator
├── results/                 # Generated outputs
│   ├── EDA/                # EDA visualizations
│   ├── ML/                 # ML pipeline results
│   └── MODEL RESULTS/      # Serialized model results
├── main.py                  # Entry point script
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- UV package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd iris_classification

# Install dependencies
uv sync

# Run the complete pipeline
python main.py
```

### Usage

```bash
# Run the complete pipeline
python main.py

# Run individual components
python src/eda_analysis.py
python src/ml_pipeline.py
python src/ml_visualizations.py

# Run Jupyter notebooks for exploration
jupyter notebook notebooks/
```

### Development Workflow

```bash
# Install development dependencies
uv sync --dev

# Run with hot reload during development
uv run python main.py
```

## 📋 Modules & Scripts

### Core Modules (`src/`)

#### `eda_analysis.py`
- Comprehensive exploratory data analysis
- Generates 4 EDA visualizations (distributions, correlations, pairwise plots, boxplots)
- Detects outliers and correlations
- Provides feature insights and recommendations

#### `ml_pipeline.py`
- End-to-end ML pipeline implementation
- Trains 4 baseline and 4 tuned models
- Performs hyperparameter optimization with GridSearchCV
- Generates detailed performance reports and saves results

#### `ml_visualizations.py`
- Creates professional ML visualizations
- 5 comprehensive analysis plots (accuracy comparison, confusion matrices, cross-validation, metrics, improvement)
- High data-ink ratio design with consistent styling

### Jupyter Notebooks (`notebooks/`)

#### `data_ingestion.ipynb`
- Data loading and validation
- Initial data quality checks
- Data preparation for analysis

#### `EDA.ipynb`
- Interactive exploratory data analysis
- Statistical analysis and insights
- Visualization prototyping

#### `ml_pipeline.ipynb`
- Interactive model training and evaluation
- Hyperparameter tuning experiments
- Results analysis and interpretation

#### `ml_vis.ipynb`
- Visualization design and refinement
- Plot styling and customization
- Figure export and formatting

## Model Insights

### Decision Tree (Best Performer)
- **Optimal Parameters**: max_depth=3, criterion='gini', min_samples_split=2
- **Strengths**: Excellent interpretability, perfect precision on Setosa/Versicolor
- **Performance**: 97.78% accuracy with only 1 misclassification
- **Recommendation**: Production-ready for balanced interpretability-performance trade-off

### Support Vector Machine (Consistent)
- **Optimal Parameters**: C=100, kernel='linear', gamma='scale'
- **Strengths**: Consistent performance, already optimal at baseline
- **Performance**: 93.33% accuracy with excellent generalization
- **Recommendation**: Good choice when margin-based decision boundaries are preferred

## Data Quality Assessment

- **Missing Values**: None
- **Outliers**: 4 outliers in sepal width (2.7%)
- **Correlations**: High correlation between petal length & width (r=0.963)
- **Class Balance**: Perfectly balanced (50/50/50)
- **Feature Scaling**: Applied StandardScaler for distance-based algorithms

## Feature Engineering Insights

From the correlation analysis:
- **Petal Length & Width**: Highly correlated (r=0.963) - most discriminative features
- **Sepal Measurements**: Lower correlation, less discriminative power
- **Class Separability**: Setosa easily separable, Versicolor/Virginica overlap
- **Scaling Impact**: Critical for SVM and Logistic Regression performance

## Business Applications

This classification pipeline can be adapted for:
- **Botanical Research**: Automated flower species identification
- **Quality Control**: Product classification based on measurements
- **Medical Diagnosis**: Multi-class disease classification
- **Customer Segmentation**: Grouping based on behavioral metrics

## Performance Metrics

### Classification Report Summary (Best Model)
```
                 precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       1.00      0.93      0.97        15
 Iris-virginica       0.94      1.00      0.97        15
    accuracy                           0.98        45
   macro avg       0.98      0.98      0.98        45
weighted avg       0.98      0.98      0.98        45
```

## 💡 Key Features & Highlights

- **🔬 Scientific Approach**: Rigorous statistical analysis and validation
- **📊 Publication-Ready Visualizations**: Professional plots with consistent styling
- **⚡ Performance Optimized**: Efficient hyperparameter tuning and model selection
- **🔄 Reproducible**: Fixed random seeds and deterministic results
- **📈 Scalable Architecture**: Modular design for easy extension
- **🧪 Experiment Tracking**: Comprehensive logging and result serialization

## 🎓 Learning Outcomes

This project demonstrates expertise in:
- **Machine Learning Pipeline Design**: End-to-end ML workflow implementation
- **Statistical Analysis**: Comprehensive EDA and feature engineering
- **Model Selection & Evaluation**: Systematic comparison of multiple algorithms
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Data Visualization**: Professional plotting and visual storytelling
- **Code Organization**: Clean, modular, and maintainable codebase
- **Reproducible Research**: Version-controlled experiments and results

## 🚀 Production Readiness

The pipeline includes production-ready features:
- **Error Handling**: Robust exception handling and logging
- **Data Validation**: Input validation and quality checks
- **Model Persistence**: Serialized model results for deployment
- **Configuration Management**: Centralized project configuration
- **Documentation**: Comprehensive code documentation and README

## 🤝 Contributing

This project serves as a portfolio showcase demonstrating:
- **ML Pipeline Design**: From raw data to production-ready models
- **Statistical Analysis**: Comprehensive EDA and insights extraction
- **Visualization Excellence**: Professional data storytelling
- **Code Quality**: Clean, documented, and reproducible code

## 📞 Contact

**Imtiaz Nabi**  
📧 [imtiaznabi8@gmail.com](mailto:imtiaznabi8@gmail.com)  
💼 [LinkedIn Profile](https://www.linkedin.com/in/imtiaz-nabi/)  
🐙 [GitHub Portfolio](https://github.com/imtiaznabi)

---

*🌸 This project demonstrates the complete lifecycle of a machine learning classification task, from data exploration to model deployment-ready solutions.*