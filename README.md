# Lung-Cancer-prediction-Modeling
LUNG CANCER PREDICTION MODELING
# Lung Cancer Prediction Capstone Project

## Group 7 Members
- **Tanvitha Doradla** - 1002060626
- **Harshkumar Vijaykumar Soni** - 1002039451
- **Obalanlege Adewale** - 1001994169

**Under the guidance of Prof. Rozsa Zaruba**

---

## Abstract
This project develops a comprehensive predictive model to identify individuals at high risk of lung cancer. Utilizing publicly available datasets, including demographic, lifestyle, and health data from thousands of individuals from the NHANES survey, various machine learning techniques were employed, such as Random Forest, XGBoost, and Support Vector Machines (SVM). The Random Forest model demonstrated robust performance with an accuracy of 92%, precision of 90%, and recall of 91%. This study aims to facilitate early detection and improve patient outcomes by identifying the most influential factors contributing to lung cancer risk. Additionally, an integrated web application with a user-friendly interface was developed to enable individuals to assess their risk and promote awareness about the importance of early detection.

## Introduction
Lung cancer is a leading cause of cancer-related deaths worldwide, with over 2 million new cases annually. The complexity of lung cancer's etiology, involving genetic, environmental, and lifestyle factors, complicates early detection and treatment. This project leverages advancements in machine learning to develop a predictive tool that addresses the critical gap in early detection, offering a significant step forward in patient prognosis and management.

## Problem Statement
Early detection of lung cancer is hindered by the lack of specific symptoms in its early stages, leading to late diagnoses and poor survival rates. This project addresses this challenge by utilizing machine learning to analyze a comprehensive range of health indicators and predict lung cancer risk, aiming to facilitate earlier diagnostic interventions and improve patient outcomes.

## Methodology
The project utilized the NHANES dataset, comprising demographic, lifestyle, and health data from thousands of individuals. Data preprocessing included handling missing values, encoding categorical variables, and normalizing features. Several machine learning algorithms were evaluated, including logistic regression, decision trees, random forests, SVM, XGBoost, and LightGBM. Models were selected based on their ability to handle imbalanced data and provide interpretable results, which are essential for medical diagnostics.

## Analysis and Results
The analysis encompassed data preprocessing, exploratory data analysis (EDA), feature engineering, dimensionality reduction using Principal Component Analysis (PCA), clustering analysis with K-Means, model selection, training, evaluation, hyperparameter tuning, and interpretation of results.

The Random Forest model outperformed other algorithms, achieving an accuracy of 92%, precision of 90%, and recall of 91%. Model validation was performed using k-fold cross-validation to ensure robustness. The results are presented through confusion matrices and ROC curves, providing clear insights into model performance and reliability.

Feature importance analysis revealed that age, coughing, shortness of breath, and chest pain were among the most influential factors contributing to lung cancer predictions. Visualizations of decision boundaries and feature importances provided insights into the behavior and decision-making process of the models.

## Web Application
A Flask-based web application was developed to integrate the machine learning model backend with a user-friendly front-end interface. This application allows users to enter their personal health data and receive an assessment of their lung cancer risk. The web application architecture includes Flask routes for handling requests and serving model predictions.

## Conclusions
The project successfully demonstrates the application of machine learning techniques in predicting lung cancer risk, with potential for real-world application in clinical settings. By integrating these models into routine screening processes, it may be possible to significantly improve early detection rates and patient outcomes.

## Lessons Learned
The project highlighted the importance of comprehensive data preprocessing and the selection of appropriate models for health data. Key lessons learned include:

1. Importance of data preprocessing, including handling missing values and addressing class imbalance.
2. Value of exploratory data analysis for gaining insights and guiding feature engineering decisions.
3. Power of feature engineering in improving model performance and understanding underlying relationships.
4. Usefulness of dimensionality reduction techniques like PCA for visualization and potential performance enhancement.
5. Significance of model selection, evaluation, and hyperparameter tuning for optimizing performance.
6. Interpretability and explainability through feature importance analysis and decision boundary visualization.
7. Reproducibility and deployment facilitated by serializing trained models.
8. Importance of effective collaboration, communication, and documentation throughout the project lifecycle.

## Future Work
Future work will focus on integrating more diverse datasets, exploring advanced machine learning techniques such as deep learning, and expanding the model to predict other types of cancers. Further development of the web application is planned to include more interactive features and personalized feedback based on individual risk factors.

## Datasets

- National Health and Nutrition Examination Survey (NHANES) Dataset. Centers for Disease Control and Prevention. [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm)
- Ahmad Bhat, M. (2021, October 1). Lung cancer [Dataset]. Kaggle. [Kaggle Lung Cancer Dataset](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)

## Tools and Libraries

- [Python Programming Language](https://www.python.org/)
- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/)
- [XGBoost: Extreme Gradient Boosting](https://xgboost.readthedocs.io/)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://lightgbm.readthedocs.io/)
- [Flask: Web Framework for Python](https://flask.palletsprojects.com/)

## References

- Aberle, D. R., et al. (2011). Reduced lung-cancer mortality with low-dose computed tomographic screening. New England Journal of Medicine, 365(5), 395-409. [doi:10.1056/NEJMoa1102873](https://doi.org/10.1056/NEJMoa1102873)
- Centers for Disease Control and Prevention. (n.d.). National Health and Nutrition Examination Survey (NHANES). [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30. [LightGBM Paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- Pallets Projects. (n.d.). Flask. [Flask](https://flask.palletsprojects.com/)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. [Scikit-learn Paper](http://jmlr.org/papers/v12/pedregosa11a.html)
- Tammemägi, M. C., et al. (2013). Selection criteria for lung-cancer screening. New England Journal of Medicine, 368(8), 728-736. [doi:10.1056/NEJMoa1211776](https://doi.org/10.1056/NEJMoa1211776)
- Siegel, R. L., Miller, K. D., & Jemal, A. (2020). Cancer statistics, 2020. CA: A Cancer Journal for Clinicians, 70(1), 7-30.
- American Cancer Society. (2024, January 29). Lung cancer statistics: How common is lung cancer? [American Cancer Society](https://www.cancer.org/cancer/types/lung-cancer/about/key-statistics.html#:~:text=The%20American%20Cancer%20Society's%20estimates,men%20and%2059%2C280%20in%20women)
