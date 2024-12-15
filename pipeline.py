# pipeline.py
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_data import generate_student_performance_data
from src.data_preprocessing import DataPreprocessor
from models.ann_model import ANNModel
from src.feature_importance import FeatureImportance
import numpy as np

def main():
    # Generate dataset if not exists
    data_path = 'data/raw/student_performance.csv'
    if not os.path.exists(data_path):
        generate_student_performance_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(data_path)
    processed_data = preprocessor.preprocess()
    
    # Determine number of classes
    num_classes = len(np.unique(processed_data['y_train']))
    
    # Train ANN Model
    ann_model = ANNModel(
        input_shape=processed_data['X_train'].shape[1], 
        num_classes=num_classes
    )
    history = ann_model.train(
        processed_data['X_train'], 
        processed_data['y_train'],
        X_val=processed_data['X_test'],
        y_val=processed_data['y_test']
    )
    
    # Evaluate Model
    y_pred = ann_model.evaluate(
        processed_data['X_test'], 
        processed_data['y_test']
    )
    
    # Feature Importance
    FeatureImportance.plot_feature_importance(
        ann_model, 
        processed_data['X_test'], 
        processed_data['y_test'],
        processed_data['feature_names']
    )
    
    # Correlation Analysis
    FeatureImportance.correlation_analysis(
        processed_data['X_test'], 
        processed_data['feature_names']
    )
    
    # Save Model
    ann_model.save_model()

if __name__ == '__main__':
    main()