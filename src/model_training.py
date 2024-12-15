# src/model_training.py
from models.ann_model import ANNModel
from src.data_preprocessing import DataPreprocessor

class GradePredictionModel:
    def __init__(self, data_path):
        self.preprocessor = DataPreprocessor(data_path)
        self.processed_data = self.preprocessor.preprocess()
        
    def train_ann_model(self):
        """
        Train Artificial Neural Network Model
        """
        ann_model = ANNModel(input_shape=self.processed_data['X_train'].shape[1])
        history = ann_model.train(
            self.processed_data['X_train'], 
            self.processed_data['y_train'],
            X_val=self.processed_data['X_test'],
            y_val=self.processed_data['y_test']
        )
        
        # Evaluate and save model
        ann_model.evaluate(
            self.processed_data['X_test'], 
            self.processed_data['y_test']
        )
        ann_model.save_model()
        
        return ann_model, history

    def train_decision_tree_model(self):
        """
        Optional: Train Decision Tree Model if needed
        """
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import classification_report
        
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(self.processed_data['X_train'], self.processed_data['y_train'])
        
        # Predict and evaluate
        y_pred = dt_model.predict(self.processed_data['X_test'])
        print(classification_report(
            self.processed_data['y_test'], 
            y_pred
        ))
        
        return dt_model