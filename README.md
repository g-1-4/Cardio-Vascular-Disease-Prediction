Cardiovascular diseases have been a leading cause of mortality in the present era; the prognosis 
of cardiovascular diseases is still tricky for physicians. Fortunately, there are sequences and patterns present 
in cardiovascular diseases. Machine Learning and Deep learning techniques can identify this pattern and 
provide valid predictions for prognosis. This research may help in predicting cardiovascular diseases and 
reduce misdiagnosis, which helps in reducing the fatality of this disease. Initial data preprocessing included 
handling outliers through Z-scores, followed by classifying the data through clustering using the K-modes 
algorithm with Huang to increase the accuracy and reliability of the model. Machine learning models include 
Catboost Classifier, Gradient Boosting Classifier, K-Nearest Neighbor Classifier (KNN), Light Gradient 
Boosting Classifier, and Support Vector Machine. We have also used Recurrent Neural Networks (RNN) and 
Long Short-Term Model (LSTM). GridSearchCV was used to hypertune the model's parameters to increase 
its accuracy and reliability. After preprocessing the data, we apply the model to 61,158 dataset instances with 
12 attributes. We train the proposed models on the data, splitting it into 60:40, 70:30, and 80:20 ratios, 
achieving the following accuracies: CatBoost: 88.11(without cross-validation) and 87.77(with cross
validation), Gradient Boosting: 87.21(without cross-validation) and 88.01(with cross-validation), KNN: 
86.88(without cross-validation) and 87.33(with cross-validation), Light Gradient Boosting: 88.09(without 
cross-validation) and 88.01(with cross-validation), SVM: 87.00(without cross-validation) and 88.09(with 
cross-validation), for Recurrent Neural Networks(RNN): 88.01 and Long Short Term Model(LSTM): 88.13. 
The Area-Under-Curve for respective models: CatBoost: 0.96, Gradient Boosting: 0.95, KNN: 0.93, Light 
Gradient Boosting: 0.96, SVM: 0.93. From the following accuracies, we can conclude that Long short-term 
model and CatBoost without cross-validation have the highest accuracy respectively at 88.13% and 88.11%. 
