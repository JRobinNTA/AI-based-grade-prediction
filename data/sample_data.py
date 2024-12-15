# data/sample_data.py
import pandas as pd
import numpy as np

def generate_student_performance_data(n_samples=1000):
    """
    Generate synthetic student performance dataset with more varied grade distribution
    """
    np.random.seed(42)
    
    # Generate features with more variance
    attendance = np.random.normal(75, 15, n_samples)  # Mean 75, std 15
    participation = np.random.normal(5, 2, n_samples)  # Mean 5, std 2
    assignment_scores = np.random.normal(75, 15, n_samples)
    midterm_exam = np.random.normal(70, 20, n_samples)
    prev_gpa = np.random.normal(3.0, 0.5, n_samples)
    
    # Clip values to realistic ranges
    attendance = np.clip(attendance, 0, 100)
    participation = np.clip(participation, 0, 10)
    assignment_scores = np.clip(assignment_scores, 0, 100)
    midterm_exam = np.clip(midterm_exam, 0, 100)
    prev_gpa = np.clip(prev_gpa, 0, 4.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'attendance': attendance,
        'participation': participation,
        'assignment_scores': assignment_scores,
        'midterm_exam': midterm_exam,
        'prev_gpa': prev_gpa
    })
    
    # More nuanced grade calculation
    def calculate_grade(row):
        weighted_score = (
            row['attendance'] * 0.2 + 
            row['participation'] * 5 + 
            row['assignment_scores'] * 0.3 + 
            row['midterm_exam'] * 0.3 + 
            row['prev_gpa'] * 10
        )
        
        if weighted_score >= 90: return 'A'
        elif weighted_score >= 80: return 'B'
        elif weighted_score >= 70: return 'C'
        elif weighted_score >= 60: return 'D'
        else: return 'F'
    
    df['final_grade'] = df.apply(calculate_grade, axis=1)
    
    # Ensure data directory exists
    import os
    os.makedirs('data/raw', exist_ok=True)
    
    # Save dataset
    output_path = 'data/raw/student_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    
    return df

# Allow script to be run directly
if __name__ == '__main__':
    generate_student_performance_data()