# Machine Learning Lecture Series
## Python Notebooks for BS Data Science Students

A comprehensive collection of Jupyter notebooks covering fundamental machine learning concepts, from introduction to supervised and unsupervised learning models, with practical examples and hands-on exercises.

---

## 📚 Overview

This repository contains a complete lecture series on machine learning designed for undergraduate Data Science students. Each notebook combines theoretical explanations with practical code examples, visualizations, and exercises to provide a thorough understanding of machine learning concepts.

---

## 📁 Repository Structure

```
ml_lecture_notebooks/
│
├── 01_Introduction_to_Machine_Learning.ipynb
├── 02_Supervised_Learning_Regression.ipynb
├── 03_Supervised_Learning_Classification.ipynb
├── 04_Unsupervised_Learning.ipynb
├── 05_Model_Evaluation_and_Selection.ipynb
├── 06_Practical_Considerations_and_Best_Practices.ipynb
├── 07_Mini_Lab_Guided_Practice.ipynb
└── README.md
```

---

## 📖 Notebook Contents

### 1. Introduction to Machine Learning
**File**: `01_Introduction_to_Machine_Learning.ipynb`

- What is Machine Learning?
- Types of Machine Learning (Supervised, Unsupervised, Reinforcement)
- Key Concepts and Terminology
- Machine Learning Workflow
- Data Splitting and Evaluation
- Overfitting and Underfitting

**Key Topics**: Features, labels, training/validation/test sets, bias-variance tradeoff, evaluation metrics

---

### 2. Supervised Learning: Regression Models
**File**: `02_Supervised_Learning_Regression.ipynb`

- Introduction to Regression
- Linear Regression (Simple and Multiple)
- Regularization Techniques (Ridge, Lasso, Elastic Net)
- Polynomial Regression
- Evaluation Metrics (MSE, RMSE, MAE, R²)

**Key Topics**: OLS, Gradient Descent, feature scaling, coefficient interpretation, regularization

**Code Examples**: 
- Simple and multiple linear regression
- Regularization comparison
- Polynomial regression with different degrees

---

### 3. Supervised Learning: Classification Models
**File**: `03_Supervised_Learning_Classification.ipynb`

- Introduction to Classification
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)

**Key Topics**: Decision boundaries, class probabilities, splitting criteria, ensemble methods, kernels

**Code Examples**:
- Logistic regression with ROC curves and confusion matrices
- Decision trees with visualization
- Random Forest feature importance
- SVM with different kernels
- k-NN with varying k values

---

### 4. Unsupervised Learning Models
**File**: `04_Unsupervised_Learning.ipynb`

- Introduction to Unsupervised Learning
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN Clustering
- Principal Component Analysis (PCA)
- t-SNE Visualization

**Key Topics**: Clustering algorithms, dimensionality reduction, evaluation metrics, variance explained

**Code Examples**:
- K-Means with elbow method and silhouette analysis
- Hierarchical clustering with dendrograms
- DBSCAN for non-spherical clusters
- PCA with scree plots and variance analysis
- t-SNE for non-linear visualization

---

### 5. Model Evaluation and Selection
**File**: `05_Model_Evaluation_and_Selection.ipynb`

- Cross-Validation (k-Fold, Stratified, LOOCV)
- Hyperparameter Tuning (Grid Search, Random Search, Bayesian Optimization)
- Model Comparison and Selection
- Bias-Variance Tradeoff
- Occam's Razor

**Key Topics**: Resampling techniques, parameter optimization, model comparison strategies

**Code Examples**:
- k-fold and stratified cross-validation
- Grid search and random search for hyperparameter tuning
- Model comparison using cross-validation

---

### 6. Practical Considerations and Best Practices
**File**: `06_Practical_Considerations_and_Best_Practices.ipynb`

- Data Preprocessing (missing values, scaling, encoding)
- Feature Engineering and Selection
- Common Pitfalls (data leakage, overfitting, class imbalance)
- Model Interpretability
- Best Practices

**Key Topics**: Preprocessing pipelines, feature importance, SHAP values, avoiding common mistakes

---

### 7. Mini-Lab: Guided Practice
**File**: `07_Mini_Lab_Guided_Practice.ipynb`

A hands-on practice notebook where students:
- Choose a dataset from provided options
- Frame supervised and unsupervised learning tasks
- Implement complete ML pipelines
- Train and evaluate models
- Apply clustering and dimensionality reduction
- Interpret results and extract insights

**Datasets Available**:
- Iris Dataset (Easy)
- Wine Quality Dataset (Medium)
- Breast Cancer Wisconsin (Medium)
- California Housing (Advanced)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see below)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ml_lecture_notebooks.git
   cd ml_lecture_notebooks
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
   ```

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

   Or with JupyterLab:
   ```bash
   jupyter lab
   ```

---

## 📦 Required Packages

The notebooks require the following Python packages:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
jupyter>=1.0.0
```

### Creating requirements.txt

A `requirements.txt` file is recommended. Create it with:

```bash
pip freeze > requirements.txt
```

Or create it manually with the versions above.

---

## 📝 Usage Instructions

### For Students

1. **Start with Notebook 1**: Begin with the introduction to understand fundamental concepts
2. **Follow Sequential Order**: Notebooks build upon each other, so follow the numerical order
3. **Complete Exercises**: Each notebook includes practice exercises - complete them for better understanding
4. **Run Code Cells**: Execute all code cells to see visualizations and results
5. **Complete Mini-Lab**: Use Notebook 7 as a comprehensive practice exercise

### For Instructors

1. **Use as Lecture Material**: Each notebook can be used as a standalone lecture
2. **Customize Content**: Feel free to modify notebooks to fit your course structure
3. **Assign Mini-Lab**: Notebook 7 can be assigned as a graded assignment
4. **Add Your Datasets**: Modify the mini-lab to include your own datasets

---

## 🎯 Learning Path

### Week 1-2: Foundations
- **Notebook 1**: Introduction to Machine Learning
- **Notebook 6**: Practical Considerations (preprocessing basics)

### Week 3-4: Supervised Learning - Regression
- **Notebook 2**: Regression Models
- **Notebook 5**: Model Evaluation (cross-validation basics)

### Week 5-6: Supervised Learning - Classification
- **Notebook 3**: Classification Models
- **Notebook 5**: Model Evaluation (continued)

### Week 7-8: Unsupervised Learning
- **Notebook 4**: Unsupervised Learning Models

### Week 9-10: Advanced Topics & Practice
- **Notebook 5**: Advanced Model Evaluation
- **Notebook 6**: Best Practices
- **Notebook 7**: Mini-Lab (Comprehensive Practice)

---

## 💡 Key Features

- ✅ **Comprehensive Theory**: Detailed explanations of concepts with mathematical foundations
- ✅ **Practical Examples**: Working code examples for every algorithm
- ✅ **Visualizations**: Rich plots and diagrams to aid understanding
- ✅ **Real-World Applications**: Examples from various domains
- ✅ **Best Practices**: Industry-standard approaches and common pitfalls
- ✅ **Hands-On Practice**: Guided mini-lab for practical application
- ✅ **Exercises**: Practice questions at the end of each notebook

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` for seaborn
```bash
pip install seaborn
```

**Issue**: Matplotlib style not found
```python
# Replace 'seaborn-v0_8-darkgrid' with 'seaborn-darkgrid' or 'default'
plt.style.use('default')
```

**Issue**: Jupyter notebook not launching
```bash
# Try installing jupyterlab instead
pip install jupyterlab
jupyter lab
```

**Issue**: Scikit-learn version compatibility
```bash
pip install --upgrade scikit-learn
```

---

## 📊 Dataset Information

### Built-in Datasets (from scikit-learn)

- **Iris**: `load_iris()` - 150 samples, 4 features, 3 classes
- **Breast Cancer**: `load_breast_cancer()` - 569 samples, 30 features, 2 classes
- **California Housing**: `fetch_california_housing()` - ~20,000 samples, 8 features

### External Datasets

- **Wine Quality**: Available via `fetch_openml()` (requires internet connection)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve these notebooks:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

---

## 📄 License

This educational material is provided for academic use. Feel free to use, modify, and distribute for educational purposes.

---

## 👨‍🏫 For Instructors

### Suggested Course Structure

- **Prerequisites**: Basic Python programming, statistics fundamentals
- **Course Duration**: 10-12 weeks (2-3 hours per week)
- **Assessment**: 
  - Weekly exercises (30%)
  - Mini-Lab assignment (30%)
  - Final project (40%)

### Customization Tips

1. **Add Your Own Examples**: Include domain-specific datasets relevant to your students
2. **Adjust Difficulty**: Modify code examples to match your students' skill level
3. **Extend Content**: Add advanced topics like neural networks, deep learning
4. **Create Solutions**: Provide solution notebooks for exercises (separate branch)

---

## 📚 Additional Resources

### Textbooks
- James, G., et al. (2013). *An Introduction to Statistical Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

### Online Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Video Resources
- Consider supplementing with video lectures from platforms like Coursera, edX, or YouTube

---

## 🐛 Reporting Issues

If you encounter any issues or have suggestions:

1. Check existing [Issues](https://github.com/yourusername/ml_lecture_notebooks/issues)
2. Create a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (Python version, package versions)

---

## ⭐ Acknowledgments

- Built using scikit-learn, pandas, matplotlib, and other open-source libraries
- Inspired by standard machine learning curricula
- Designed for educational purposes

---

## 📧 Contact

For questions or feedback:
- Open an issue on GitHub
- Contact the course instructor

---

## 🔄 Version History

- **v1.0.0** (Current)
  - Initial release
  - 7 comprehensive notebooks
  - Complete theoretical content and code examples
  - Mini-lab practice exercise

---

## 📌 Quick Start Checklist

- [ ] Clone the repository
- [ ] Install required packages
- [ ] Launch Jupyter Notebook
- [ ] Open `01_Introduction_to_Machine_Learning.ipynb`
- [ ] Run the first code cell to verify installation
- [ ] Start learning! 🎓

---

**Happy Learning!** 🚀

*Last Updated: December 2025*

