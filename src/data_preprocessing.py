# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data_path, target_column=None, drop_columns=None):
        """
        Initialize preprocessor with flexible data handling
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file
        target_column : str, optional
            Name of the target/label column
        drop_columns : list, optional
            List of columns to drop before processing
        """
        # Read the data
        self.data = pd.read_csv(data_path)
        
        # Drop specified columns if provided
        if drop_columns:
            self.data = self.data.drop(columns=drop_columns)
        
        # Identify target column
        if target_column is None:
            # Assume last column is target if not specified
            self.target_column = self.data.columns[-1]
        else:
            self.target_column = target_column
        
        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]
        
        # Initialize placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def preprocess(self, test_size=0.2, random_state=42):
        """
        Comprehensive data preprocessing pipeline
        
        Parameters:
        -----------
        test_size : float, optional
            Proportion of data to use for testing
        random_state : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        dict: Processed data and metadata
        """
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(self.X)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Encode target variable if it's categorical
        if self.y.dtype == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(self.y)
        else:
            label_encoder = None
            y_encoded = self.y
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': list(self.X.columns),
            'label_encoder': label_encoder,
            'target_column': self.target_column
        }
    
    def get_data_info(self):
        """
        Provide basic information about the dataset
        """
        print("Dataset Information:")
        print(f"Total Columns: {len(self.data.columns)}")
        print(f"Target Column: {self.target_column}")
        print("\nColumn Details:")
        for col in self.data.columns:
            print(f"{col}: {self.data[col].dtype}")
        
        return self.data.info()