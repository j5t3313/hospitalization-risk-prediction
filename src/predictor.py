import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional


class HospitalizationRiskPredictor:
    """
    Hospitalization risk prediction system.
    
    This class provides an interface for predicting hospitalization risk
    using a trained Gradient Boosting model. It supports both single-patient
    and batch predictions with configurable decision thresholds.
    
    Attributes:
        model: Trained scikit-learn model
        scaler: Feature scaler (StandardScaler) if applicable
        feature_cols: List of required feature names
        model_name: Name/type of the trained model
        threshold: Decision threshold for high-risk classification
        performance: Dictionary of model performance metrics
        threshold_analysis: Optional DataFrame with threshold trade-off analysis
    
    Example:
        >>> predictor = HospitalizationRiskPredictor(
        ...     'model_file.pkl', 
        ...     threshold=0.20
        ... )
        >>> result = predictor.predict(patient_data)
        >>> print(result['probability'])
        0.34
    """
    
    def __init__(self, model_path: str, threshold: Optional[float] = None):
        """
        Initialize the predictor by loading a trained model.
        
        Args:
            model_path: Path to the pickled model file (.pkl)
            threshold: Decision threshold for binary classification.
                      If None, uses the recommended threshold from model package.
                      
        Raises:
            FileNotFoundError: If model_path does not exist
            KeyError: If model package is missing required components
        """
        with open(model_path, 'rb') as f:
            package = pickle.load(f)
        
        self.model = package['model']
        self.scaler = package.get('scaler', None)
        self.feature_cols = package['feature_columns']
        self.model_name = package['model_name']
        self.threshold = threshold if threshold is not None else package.get('recommended_threshold', 0.2)
        self.performance = package['performance']
        self.threshold_analysis = package.get('threshold_analysis', None)
    
    def predict(self, patient_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Predict hospitalization risk for a single patient.
        
        Args:
            patient_data: Dictionary or single-row DataFrame containing 
                         patient features. Must include all required features.
        
        Returns:
            Dictionary containing:
                - probability (float): Predicted probability of hospitalization (0-1)
                - high_risk_flag (int): Binary classification (1=high risk, 0=low risk)
                - risk_level (str): Categorical risk level (Low/Moderate/High/Very High)
                - model (str): Model name used for prediction
                - threshold (float): Decision threshold used
        
        Example:
            >>> patient = {
            ...     'age': 78,
            ...     'num_chronic_conditions': 4,
            ...     'baseline_hospital_admits': 1.0,
            ...     # ... other features
            ... }
            >>> result = predictor.predict(patient)
            >>> print(f"Risk: {result['probability']:.1%}")
            Risk: 34.1%
        """
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Validate features
        missing_features = set(self.feature_cols) - set(patient_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = patient_df[self.feature_cols]
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Generate prediction
        probability = self.model.predict_proba(X)[0, 1]
        high_risk = int(probability >= self.threshold)
        
        # Determine risk level category
        if probability < 0.15:
            risk_level = 'Low Risk'
        elif probability < 0.25:
            risk_level = 'Moderate Risk'
        elif probability < 0.35:
            risk_level = 'High Risk'
        else:
            risk_level = 'Very High Risk'
        
        return {
            'probability': float(probability),
            'high_risk_flag': high_risk,
            'risk_level': risk_level,
            'model': self.model_name,
            'threshold': self.threshold
        }
    
    def batch_predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Predict hospitalization risk for multiple patients.
        
        Args:
            dataframe: DataFrame containing patient features. Must include
                      all required features and can include additional columns.
        
        Returns:
            Copy of input DataFrame with added columns:
                - hospitalization_probability: Predicted probability (0-1)
                - high_risk_flag: Binary classification (1=high risk, 0=low risk)
                - risk_level: Categorical risk level
        
        Example:
            >>> patients_df = pd.read_csv('patients.csv')
            >>> predictions = predictor.batch_predict(patients_df)
            >>> print(predictions[['patient_id', 'hospitalization_probability', 'risk_level']])
        """
        # Validate features
        missing_features = set(self.feature_cols) - set(dataframe.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = dataframe[self.feature_cols]
        
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Generate predictions
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Create results DataFrame
        results = dataframe.copy()
        results['hospitalization_probability'] = probabilities
        results['high_risk_flag'] = (probabilities >= self.threshold).astype(int)
        results['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.15, 0.25, 0.35, 1.0],
            labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
        )
        
        return results
    
    def get_feature_requirements(self) -> List[str]:
        """
        Return list of required feature names for prediction.
        
        Returns:
            List of feature column names required by the model
        
        Example:
            >>> features = predictor.get_feature_requirements()
            >>> print(f"Model requires {len(features)} features")
            Model requires 36 features
        """
        return self.feature_cols.copy()
    
    def get_model_info(self) -> Dict:
        """
        Return model metadata and configuration.
        
        Returns:
            Dictionary containing:
                - model_name: name/type of model
                - threshold: current decision threshold
                - performance: performance metrics dict
                - n_features: number of required features
                - has_threshold_analysis: whether threshold tradeoff data available
        
        Example:
            >>> info = predictor.get_model_info()
            >>> print(f"Model: {info['model_name']}")
            >>> print(f"ROC-AUC: {info['performance']['roc_auc']:.4f}")
        """
        return {
            'model_name': self.model_name,
            'threshold': self.threshold,
            'performance': self.performance,
            'n_features': len(self.feature_cols),
            'has_threshold_analysis': self.threshold_analysis is not None
        }
    
    def set_threshold(self, new_threshold: float) -> None:
        """
        Update the decision threshold for binary classification.
        
        Args:
            new_threshold: New threshold value (0-1). Common values:
                          0.15 (high recall), 0.20 (balanced), 0.25 (high precision)
        
        Raises:
            ValueError: If threshold not in valid range [0, 1]
        
        Example:
            >>> predictor.set_threshold(0.15)  # Increase recall
            Threshold updated to: 0.15
            Expected performance at this threshold:
              Precision: 0.2600
              Recall: 0.8364
              F1 Score: 0.3967
              % Population Flagged: 52.1%
        """
        if not 0 <= new_threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {new_threshold}")
        
        self.threshold = new_threshold
        print(f"Threshold updated to: {new_threshold:.2f}")
        
        # If threshold analysis available, show expected performance
        if self.threshold_analysis:
            matching_analysis = [
                t for t in self.threshold_analysis 
                if abs(t['threshold'] - new_threshold) < 0.01
            ]
            if matching_analysis:
                analysis = matching_analysis[0]
                print(f"Expected performance at this threshold:")
                print(f"  Precision: {analysis['precision']:.4f}")
                print(f"  Recall: {analysis['recall']:.4f}")
                print(f"  F1 Score: {analysis['f1_score']:.4f}")
                print(f"  % Population Flagged: {analysis['flagged_pct']:.1f}%")
    
    def get_threshold_options(self) -> Optional[pd.DataFrame]:
        """
        Return threshold analysis data if available.
        
        Returns:
            DataFrame with columns: threshold, precision, recall, f1_score, flagged_pct
            Returns None if threshold analysis not available in model package
        
        Example:
            >>> options = predictor.get_threshold_options()
            >>> print(options[['threshold', 'recall', 'precision']])
               threshold  recall  precision
            0       0.10  0.9283     0.2368
            1       0.15  0.8364     0.2600
            2       0.20  0.6691     0.2971
        """
        if self.threshold_analysis:
            return pd.DataFrame(self.threshold_analysis)
        return None
    
    def predict_with_explanation(self, patient_data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Predict with additional context about the prediction.
        
        Args:
            patient_data: Dictionary or single-row DataFrame containing patient features
        
        Returns:
            Dictionary containing standard prediction fields plus:
                - feature_values: Dict of input feature values
                - top_risk_factors: List of features contributing most to risk (only available for tree-based models)
        
        Note:
            Feature importance explanation only available for models with feature_importances_ attribute (Random Forest, Gradient Boosting)
        """
        result = self.predict(patient_data)
        
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
        
        # Add feature values to result
        result['feature_values'] = patient_df[self.feature_cols].iloc[0].to_dict()
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            result['top_risk_factors'] = importances.head(5)['feature'].tolist()
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the predictor."""
        return (
            f"HospitalizationRiskPredictor("
            f"model='{self.model_name}', "
            f"threshold={self.threshold:.2f}, "
            f"n_features={len(self.feature_cols)})"
        )


def load_predictor(model_path: str, threshold: Optional[float] = None) -> HospitalizationRiskPredictor:
    """
    Convenience function to load a predictor.
    
    Args:
        model_path: Path to pickled model file
        threshold: Optional decision threshold
    
    Returns:
        Initialized HospitalizationRiskPredictor instance
    
    Example:
        >>> predictor = load_predictor('models/model.pkl', threshold=0.20)
    """
    return HospitalizationRiskPredictor(model_path, threshold)


if __name__ == "__main__":
    # Example usage
    print("Hospitalization Risk Predictor")
    print("=" * 50)
    print("\nThis module provides the HospitalizationRiskPredictor class")
    print("for predicting 30-day hospitalization risk in senior living populations.")
    print("\nBasic usage:")
    print("  from predictor import HospitalizationRiskPredictor")
    print("  predictor = HospitalizationRiskPredictor('model.pkl')")
    print("  result = predictor.predict(patient_data)")
    print("\nFor more information, see the README.md file.")