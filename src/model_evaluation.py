import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np

class GradePredictionModel:
    def __init__(self):
        self.ann_model = None
        self.decision_tree = None
    
    def build_ann(self, input_shape):
        """
        Build Artificial Neural Network for regression
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_ann(self, X_train, y_train, epochs=100):
        """
        Train ANN model
        """
        self.ann_model = self.build_ann(X_train.shape[1])
        self.ann_model.fit(X_train, y_train, epochs=epochs, verbose=0)
        return self
    
    def train_decision_tree(self, X_train, y_train):
        """
        Train Decision Tree for regression
        """
        self.decision_tree = DecisionTreeRegressor(random_state=42)
        self.decision_tree.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate both models
        """
        results = {}
        
        # ANN Evaluation
        ann_predictions = self.ann_model.predict(X_test).flatten()
        results['ann_rmse'] = np.sqrt(mean_squared_error(y_test, ann_predictions))
        
        # Decision Tree Evaluation
        dt_predictions = self.decision_tree.predict(X_test)
        results['dt_rmse'] = np.sqrt(mean_squared_error(y_test, dt_predictions))
        
        return results