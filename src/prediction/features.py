"""Feature engineering for duration prediction."""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ResourceMetadata:
    """Metadata about a resource for feature extraction."""
    resource_id: str
    name: str
    role: str
    capabilities: List[str]
    hire_date: Optional[str] = None
    working_hours: Optional[str] = None


class DurationFeatureExtractor:
    """
    Extract features for duration prediction.
    
    Features (excluding time-based features for now):
    1. Experience profile statistics (mean, std, count, trend)
    2. Resource metadata (role, capability count, tenure)
    3. Context attributes (one-hot encoded)
    4. Activity characteristics (name one-hot encoded)
    
    Time-based features (day of week, time of day) excluded for now
    but can be added later for correlation analysis with performance.
    """
    
    def __init__(
        self,
        context_attributes: Optional[List[str]] = None,
        resource_metadata: Optional[Dict[str, ResourceMetadata]] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            context_attributes: List of context attribute names to include
            resource_metadata: Dictionary mapping resource_id to metadata
        """
        self.context_attributes = context_attributes or []
        self.resource_metadata = resource_metadata or {}
        
        # Fitted vocabularies (populated during fit())
        self.activity_vocab: List[str] = []
        self.context_vocab: Dict[str, List[str]] = {}
        self.role_vocab: List[str] = []
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'DurationFeatureExtractor':
        """
        Fit feature extractor to training data.
        
        Learns vocabularies for categorical encoding.
        
        Args:
            df: Training dataframe with columns:
                - activity_name
                - resource_id
                - context columns (if context_attributes specified)
        
        Returns:
            self
        """
        # Activity vocabulary
        self.activity_vocab = sorted(df['activity_name'].unique())
        
        # Context vocabularies
        for attr in self.context_attributes:
            if attr in df.columns:
                self.context_vocab[attr] = sorted(df[attr].dropna().unique())
        
        # Role vocabulary from metadata
        if self.resource_metadata:
            roles = {meta.role for meta in self.resource_metadata.values()}
            self.role_vocab = sorted(roles)
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        resource_id: str,
        activity_name: str,
        context: Dict[str, Any],
        experience_profile: Optional[Any] = None
    ) -> np.ndarray:
        """
        Transform inputs into feature vector.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            context: Case context attributes
            experience_profile: Optional ExperienceProfile for statistics
        
        Returns:
            Feature vector as numpy array
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform()")
        
        features = []
        
        # 1. Experience profile features
        if experience_profile is not None:
            features.extend([
                experience_profile.mean_duration if experience_profile.mean_duration else 0.0,
                experience_profile.std_duration if experience_profile.std_duration else 0.0,
                experience_profile.median_duration if experience_profile.median_duration else 0.0,
                np.log1p(experience_profile.count) if experience_profile.count else 0.0,  # Log transform
                experience_profile.success_rate if experience_profile.success_rate else 1.0,
                experience_profile.trend_slope if experience_profile.trend_slope else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Defaults
        
        # 2. Resource metadata features
        if resource_id in self.resource_metadata:
            meta = self.resource_metadata[resource_id]
            
            # Role one-hot encoding
            role_encoded = [1.0 if role == meta.role else 0.0 for role in self.role_vocab]
            features.extend(role_encoded)
            
            # Capability count
            features.append(float(len(meta.capabilities)))
            
            # Tenure (days since hire) - if available
            if meta.hire_date:
                try:
                    hire_date = pd.to_datetime(meta.hire_date)
                    reference_date = pd.Timestamp('2017-01-01')  # BPIC 2017 start
                    tenure_days = (reference_date - hire_date).days
                    features.append(float(max(0, tenure_days)))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            # No metadata: zero features
            features.extend([0.0] * (len(self.role_vocab) + 2))
        
        # 3. Activity one-hot encoding
        activity_encoded = [1.0 if act == activity_name else 0.0 for act in self.activity_vocab]
        features.extend(activity_encoded)
        
        # 4. Context one-hot encoding
        for attr in self.context_attributes:
            if attr in self.context_vocab:
                value = context.get(attr)
                context_encoded = [1.0 if val == value else 0.0 for val in self.context_vocab[attr]]
                features.extend(context_encoded)
        
        return np.array(features, dtype=np.float32)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        experience_profiles: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Fit and transform training data in one step.
        
        Args:
            df: Training dataframe with columns:
                - resource_id
                - activity_name
                - context columns
            experience_profiles: Optional dict mapping (resource_id, activity, context_key) to profiles
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        self.fit(df)
        
        features_list = []
        for _, row in df.iterrows():
            resource_id = row['resource_id']
            activity_name = row['activity_name']
            
            # Extract context
            context = {attr: row.get(attr) for attr in self.context_attributes if attr in row}
            
            # Get experience profile if available
            exp_profile = None
            if experience_profiles:
                # Try to find matching profile
                for key, profile in experience_profiles.items():
                    if key[0] == resource_id and key[1] == activity_name:
                        exp_profile = profile
                        break
            
            features = self.transform(resource_id, activity_name, context, exp_profile)
            features_list.append(features)
        
        return np.array(features_list)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        names = [
            'exp_mean_duration',
            'exp_std_duration',
            'exp_median_duration',
            'exp_log_count',
            'exp_success_rate',
            'exp_trend_slope',
        ]
        
        # Role features
        names.extend([f'role_{role}' for role in self.role_vocab])
        
        # Resource features
        names.extend(['capability_count', 'tenure_days'])
        
        # Activity features
        names.extend([f'activity_{act}' for act in self.activity_vocab])
        
        # Context features
        for attr in self.context_attributes:
            if attr in self.context_vocab:
                names.extend([f'{attr}_{val}' for val in self.context_vocab[attr]])
        
        return names
    
    def get_feature_count(self) -> int:
        """Get expected number of features."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        count = 6  # Experience features
        count += len(self.role_vocab)  # Role one-hot
        count += 2  # capability_count, tenure_days
        count += len(self.activity_vocab)  # Activity one-hot
        
        for attr in self.context_attributes:
            if attr in self.context_vocab:
                count += len(self.context_vocab[attr])
        
        return count
