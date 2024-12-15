# app/streamlit_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_model_and_preprocessors():
    """Load saved model, scaler, and label encoder"""
    # Ensure all required files exist
    required_files = [
        'models/ann_grade_predictor.h5', 
        'preprocessors/scaler.pkl', 
        'preprocessors/label_encoder.pkl'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"Required file not found: {file}")
            st.stop()
    
    try:
        model = tf.keras.models.load_model('models/ann_grade_predictor.h5')
        scaler = joblib.load('preprocessors/scaler.pkl')
        label_encoder = joblib.load('preprocessors/label_encoder.pkl')
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading preprocessors: {e}")
        st.stop()

def predict_grade(model, scaler, label_encoder, features):
    """Predict grade using loaded model"""
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    predicted_class = np.argmax(prediction, axis=1)
    return label_encoder.inverse_transform(predicted_class)[0]

def main():
    st.title('Student Grade Predictor')
    
    # Input features with default values and help text
    attendance = st.slider('Attendance (%)', 
                            min_value=0, 
                            max_value=100, 
                            value=75, 
                            help="Percentage of classes attended")
    
    participation = st.slider('Participation Score', 
                               min_value=0, 
                               max_value=10, 
                               value=5, 
                               help="Active participation in class (0-10)")
    
    assignment_scores = st.slider('Assignment Scores', 
                                   min_value=0, 
                                   max_value=100, 
                                   value=75, 
                                   help="Average score on assignments")
    
    midterm_exam = st.slider('Midterm Exam Score', 
                              min_value=0, 
                              max_value=100, 
                              value=75, 
                              help="Score achieved in midterm examination")
    
    prev_gpa = st.slider('Previous Semester GPA', 
                          min_value=0.0, 
                          max_value=4.0, 
                          value=3.0, 
                          step=0.1, 
                          help="Grade Point Average from previous semester")
    
    if st.button('Predict Grade'):
        try:
            model, scaler, label_encoder = load_model_and_preprocessors()
            features = [
                attendance, participation, assignment_scores, 
                midterm_exam, prev_gpa
            ]
            predicted_grade = predict_grade(model, scaler, label_encoder, features)
            st.success(f'Predicted Grade: {predicted_grade}')
        except Exception as e:
            st.error(f'Error in prediction: {e}')

if __name__ == '__main__':
    main()