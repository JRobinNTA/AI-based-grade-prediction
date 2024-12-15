# src/feature_importance.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

class FeatureImportance:
    @staticmethod
    def plot_feature_importance(model, X_test, y_test, feature_names):
        """
        Analyze feature importance for Keras model
        """
        # Predict using the model
        y_pred = model.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate base accuracy
        base_accuracy = accuracy_score(y_test, y_pred_classes)
        
        # Perform feature importance by shuffling
        feature_importances = np.zeros(len(feature_names))
        
        for i in range(len(feature_names)):
            # Create a copy of test data
            X_test_shuffled = X_test.copy()
            
            # Shuffle a single feature
            np.random.shuffle(X_test_shuffled[:, i])
            
            # Predict with shuffled data
            y_pred_shuffled = model.model.predict(X_test_shuffled)
            y_pred_shuffled_classes = np.argmax(y_pred_shuffled, axis=1)
            
            # Calculate accuracy drop
            shuffled_accuracy = accuracy_score(y_test, y_pred_shuffled_classes)
            feature_importances[i] = base_accuracy - shuffled_accuracy
        
        # Sort feature importances
        indices = np.argsort(feature_importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_names)), feature_importances[indices])
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel("Features")
        plt.ylabel("Importance (Accuracy Drop)")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Print feature importances
        print("Feature Importances:")
        for f, idx in enumerate(indices):
            print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")
        
        return feature_importances

    @staticmethod
    def correlation_analysis(X, feature_names):
        """
        Perform correlation analysis
        """
        # Create correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    xticklabels=feature_names, 
                    yticklabels=feature_names)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        return corr_matrix