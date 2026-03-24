"""
Data loader module for UrbanSound8K dataset.
Handles dataset ingestion, validation, and preprocessing.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

from src.utils import setup_logging, set_seed


class UrbanSoundDataset:
    """
    Dataset loader and validator for UrbanSound8K.
    
    Handles metadata loading, file validation, and quality checks
    for audio files in the UrbanSound8K dataset.
    """
    
    def __init__(self, data_root: str):
        """
        Initialize the dataset loader.
        
        Args:
            data_root: Root directory of UrbanSound8K dataset (e.g., "data/UrbanSound8K")
        """
        setup_logging()
        self.logger = logging.getLogger(__name__)
        self.data_root = Path(data_root)
        self.logger.info(f"Initialized UrbanSoundDataset with root: {self.data_root}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load the UrbanSound8K metadata CSV file.
        
        Returns:
            DataFrame containing the metadata unchanged
        """
        metadata_path = self.data_root / "metadata" / "UrbanSound8K.csv"
        self.logger.info(f"Loading metadata from: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        self.logger.info(f"Total metadata rows: {len(df)}")
        
        return df
    
    def validate_files(self) -> pd.DataFrame:
        """
        Validate audio files and filter out corrupt or invalid files.
        
        Performs the following checks for each audio file:
        - File existence
        - Audio loading (detects corrupt files)
        - Duration > 0
        - RMS energy >= 1e-4 (detects silent/near-silent files)
        
        Returns:
            DataFrame containing only valid audio file entries
        """
        df = self.load_metadata()
        valid_indices = []
        rejected_count = 0
        
        self.logger.info("Starting file validation...")
        
        for idx, row in df.iterrows():
            fold = row['fold']
            slice_file_name = row['slice_file_name']
            
            # Construct full audio path: audio/fold{fold}/{slice_file_name}
            audio_path = self.data_root / "audio" / f"fold{fold}" / slice_file_name
            
            try:
                # Check if file exists
                if not audio_path.exists():
                    self.logger.warning(f"File not found: {audio_path}")
                    rejected_count += 1
                    continue
                
                # Attempt to load audio safely
                try:
                    audio, sr = librosa.load(audio_path, sr=None)
                except Exception as e:
                    self.logger.warning(f"Failed to load audio {audio_path}: {e}")
                    rejected_count += 1
                    continue
                
                # Check duration > 0
                duration = len(audio) / sr
                if duration <= 0:
                    self.logger.warning(f"Invalid duration ({duration}s) for file: {audio_path}")
                    rejected_count += 1
                    continue
                
                # Check RMS energy >= 1e-4
                rms_energy = np.sqrt(np.mean(audio ** 2))
                if rms_energy < 1e-4:
                    self.logger.warning(f"Low RMS energy ({rms_energy:.2e}) for file: {audio_path}")
                    rejected_count += 1
                    continue
                
                # All checks passed
                valid_indices.append(idx)
                
            except Exception as e:
                # Catch any unexpected errors and continue processing
                self.logger.error(f"Unexpected error processing {audio_path}: {e}")
                rejected_count += 1
                continue
        
        # Create dataframe with only valid rows
        valid_df = df.loc[valid_indices].reset_index(drop=True)
        
        # Log summary statistics
        self.logger.info(f"Validation complete:")
        self.logger.info(f"  Total valid files: {len(valid_df)}")
        self.logger.info(f"  Total rejected files: {rejected_count}")
        self.logger.info(f"  Rejection rate: {rejected_count / len(df) * 100:.2f}%")
        
        return valid_df
    
    def create_splits(
        self,
        df: pd.DataFrame,
        output_dir: str = "data/splits",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> None:
        """
        Create deterministic stratified train/validation/test splits.
        
        Performs stratified splitting based on class labels to maintain
        class distribution across all splits. Validates class balance and
        saves splits to CSV files.
        
        Args:
            df: DataFrame containing the dataset
            output_dir: Directory to save split CSV files (default: "data/splits")
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            seed: Random seed for reproducibility (default: 42)
        """
        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        
        # Set random seed for reproducibility
        set_seed(seed)
        
        self.logger.info(f"Creating stratified splits with ratios - train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")
        self.logger.info(f"Original dataset size: {len(df)}")
        
        # Calculate original class distribution
        original_dist = df['class'].value_counts(normalize=True).sort_index() * 100
        self.logger.info("Original class distribution (%):\n" + str(original_dist.to_dict()))
        
        # First split: train vs temp (val + test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df['class'],
            random_state=seed
        )
        
        # Second split: temp into validation and test
        # Adjust test_size to get correct proportions
        test_size_adjusted = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size_adjusted,
            stratify=temp_df['class'],
            random_state=seed
        )
        
        # Log split sizes
        self.logger.info(f"Split sizes - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
        
        # Validate class balance
        train_dist = train_df['class'].value_counts(normalize=True).sort_index() * 100
        val_dist = val_df['class'].value_counts(normalize=True).sort_index() * 100
        test_dist = test_df['class'].value_counts(normalize=True).sort_index() * 100
        
        self.logger.info("Train class distribution (%):\n" + str(train_dist.to_dict()))
        self.logger.info("Validation class distribution (%):\n" + str(val_dist.to_dict()))
        self.logger.info("Test class distribution (%):\n" + str(test_dist.to_dict()))
        
        # Check for class imbalance (deviation > ±10%)
        for class_label in original_dist.index:
            orig_pct = original_dist[class_label]
            
            # Check train split
            train_pct = train_dist.get(class_label, 0)
            if abs(train_pct - orig_pct) > 10.0:
                self.logger.warning(
                    f"Class {class_label} in train split deviates by {train_pct - orig_pct:.2f}% "
                    f"from original ({train_pct:.2f}% vs {orig_pct:.2f}%)"
                )
            
            # Check validation split
            val_pct = val_dist.get(class_label, 0)
            if abs(val_pct - orig_pct) > 10.0:
                self.logger.warning(
                    f"Class {class_label} in validation split deviates by {val_pct - orig_pct:.2f}% "
                    f"from original ({val_pct:.2f}% vs {orig_pct:.2f}%)"
                )
            
            # Check test split
            test_pct = test_dist.get(class_label, 0)
            if abs(test_pct - orig_pct) > 10.0:
                self.logger.warning(
                    f"Class {class_label} in test split deviates by {test_pct - orig_pct:.2f}% "
                    f"from original ({test_pct:.2f}% vs {orig_pct:.2f}%)"
                )
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created output directory: {output_path}")
        
        # Save splits to CSV files
        train_file = output_path / "train.csv"
        val_file = output_path / "val.csv"
        test_file = output_path / "test.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        self.logger.info(f"Saved train split to: {train_file}")
        self.logger.info(f"Saved validation split to: {val_file}")
        self.logger.info(f"Saved test split to: {test_file}")
        self.logger.info("Split creation complete!")
