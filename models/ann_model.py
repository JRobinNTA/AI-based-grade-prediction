# models/ann_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class ANNModel:
    def __init__(self, input_shape, num_classes):
        self.model = self._build_model(input_shape, num_classes)
        
    def _build_model(self, input_shape, num_classes):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        # Predict probabilities
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def save_model(self, filepath='models/ann_grade_predictor.h5'):
        self.model.save(filepath)