# AI-Based Grade Prediction System

## Problem Statement

The goal of this project is to develop a machine learning system that can predict a student's final grade based on various academic performance indicators.

The specific challenge is to build a predictive model that can:
- Analyze multiple input features
- Provide accurate grade predictions
- Identify the most important factors contributing to academic performance

## Approach

### Data Generation
- Created a synthetic dataset simulating student performance
- Features include:
  - Attendance percentage
  - Participation scores
  - Assignment scores
  - Midterm exam marks
  - Previous semester GPA

### Preprocessing
- Implemented comprehensive data preprocessing pipeline
- Steps included:
  - Handling missing values
  - Feature scaling
  - Label encoding of target variable
  - Train-test splitting

### Model Development
- Developed an Artificial Neural Network (ANN) model
- Model Architecture:
  - Input layer with feature dimensions
  - Multiple hidden layers with ReLU activation
  - Dropout layers to prevent overfitting
  - Softmax output layer for multi-class classification

### Feature Importance Analysis
- Utilized SHAP (SHapley Additive exPlanations) values
- Identified key contributors to grade prediction
- Generated visual representation of feature impacts

### Web Application
- Created Streamlit-based interactive interface
- Allows users to input student details
- Provides real-time grade prediction

## Results

### Model Performance
- Metrics:
  - Accuracy: [TO BE TESTED]
  - Classification Report: Precision, Recall, F1-Score
  - Confusion Matrix visualization

### Feature Importance Insights
- Visualized key factors influencing grade prediction
- Potential insights into academic success drivers

## Challenges and Solutions


1. **Model Complexity**
   - Challenge: Balancing model complexity and interpretability
   - Solution: 
     - Used dropout layers
     - Implemented feature importance analysis
     - Created modular, easily adjustable model architecture

2. **Overfitting Prevention**
   - Challenge: Ensuring model generalizes well
   - Solutions:
     - Applied dropout layers
     - Used validation split during training
     - Implemented regularization techniques


## Setup and Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/JRobinNTA/AI-based-grade-prediction.git

# Navigate to project directory
cd grade-prediction-project

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/sample_data.py

# Run pipeline
python pipeline.py

# Launch web application
streamlit run app/streamlit_app.py
```
The correlation heatmap and the feature importance graph will be available in the project directory

## Future Improvements
- Integrate real-world dataset
- Make the model capable of identifying relevant and irrelevant   columns in a dataset
- Experiment with ensemble methods
- Add more sophisticated feature engineering
- Implement more advanced model architectures
