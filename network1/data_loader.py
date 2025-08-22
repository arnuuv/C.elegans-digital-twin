import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class C_elegansDataLoader:
    """
    Data loader for C. elegans connectome data.
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the C. elegans data
        """
        self.data_dir = data_dir
        self.connectome_data = None
        self.neuron_types = None
        self.neuron_relatedness = None
        
    def load_connectome_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load connectome data from Excel file.
        
        Args:
            file_path: Path to the connectome data file
            
        Returns:
            DataFrame containing connectome data
        """
        if file_path is None:
            # Try to find connectome data automatically
            possible_files = [
                "connectivity-data-download.xls",
                "motor-data-connectivity.xls",
                "NeuronConnectFormatted.xls"
            ]
            
            for file in possible_files:
                full_path = os.path.join(self.data_dir, file)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
        
        if file_path is None or not os.path.exists(file_path):
            raise FileNotFoundError("Could not find connectome data file")
        
        try:
            # Try to read as Excel file
            if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                self.connectome_data = pd.read_excel(file_path)
            else:
                # Try to read as CSV
                self.connectome_data = pd.read_csv(file_path)
            
            print(f"Loaded connectome data: {self.connectome_data.shape}")
            return self.connectome_data
            
        except Exception as e:
            print(f"Error loading connectome data: {e}")
            return None
    
    def load_neuron_types(self, file_path: str = None) -> pd.DataFrame:
        """
        Load neuron type classification data.
        
        Args:
            file_path: Path to the neuron type file
            
        Returns:
            DataFrame containing neuron type data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "Neuron-type.xls")
        
        if not os.path.exists(file_path):
            print(f"Neuron type file not found: {file_path}")
            return None
        
        try:
            self.neuron_types = pd.read_excel(file_path)
            print(f"Loaded neuron types: {self.neuron_types.shape}")
            return self.neuron_types
        except Exception as e:
            print(f"Error loading neuron types: {e}")
            return None
    
    def load_neuron_relatedness(self, file_path: str = None) -> pd.DataFrame:
        """
        Load neuron relatedness data.
        
        Args:
            file_path: Path to the neuron relatedness file
            
        Returns:
            DataFrame containing neuron relatedness data
        """
        if file_path is None:
            # Try to find relatedness data
            possible_files = [
                "Neuron-relatedness-part1.xls",
                "Neuron-relatedness-part2.xls"
            ]
            
            for file in possible_files:
                full_path = os.path.join(self.data_dir, file)
                if os.path.exists(full_path):
                    file_path = full_path
                    break
        
        if file_path is None or not os.path.exists(file_path):
            print("Neuron relatedness file not found")
            return None
        
        try:
            self.neuron_relatedness = pd.read_excel(file_path)
            print(f"Loaded neuron relatedness: {self.neuron_relatedness.shape}")
            return self.neuron_relatedness
        except Exception as e:
            print(f"Error loading neuron relatedness: {e}")
            return None
    
    def preprocess_connectome_data(self, threshold: float = 0.0) -> np.ndarray:
        """
        Preprocess connectome data for neural network input.
        
        Args:
            threshold: Minimum connection strength threshold
            
        Returns:
            Preprocessed connectome matrix
        """
        if self.connectome_data is None:
            raise ValueError("No connectome data loaded. Call load_connectome_data() first.")
        
        # Convert to numeric matrix
        numeric_data = self.connectome_data.select_dtypes(include=[np.number])
        
        # Remove rows/columns with all zeros
        numeric_data = numeric_data.loc[(numeric_data != 0).any(axis=1)]
        numeric_data = numeric_data.loc[:, (numeric_data != 0).any(axis=0)]
        
        # Apply threshold
        if threshold > 0:
            numeric_data = (numeric_data >= threshold).astype(float)
        
        # Normalize data
        numeric_data = (numeric_data - numeric_data.mean()) / (numeric_data.std() + 1e-8)
        
        return numeric_data.values
    
    def create_adjacency_matrix(self, symmetric: bool = True) -> np.ndarray:
        """
        Create adjacency matrix from connectome data.
        
        Args:
            symmetric: Whether to make the matrix symmetric
            
        Returns:
            Adjacency matrix
        """
        if self.connectome_data is None:
            raise ValueError("No connectome data loaded. Call load_connectome_data() first.")
        
        # Get numeric data
        numeric_data = self.connectome_data.select_dtypes(include=[np.number])
        
        # Create adjacency matrix
        adj_matrix = numeric_data.values
        
        if symmetric:
            # Make symmetric by taking max of (i,j) and (j,i)
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        
        return adj_matrix
    
    def extract_features(self, feature_type: str = 'connectivity') -> np.ndarray:
        """
        Extract features from the data for neural network training.
        
        Args:
            feature_type: Type of features to extract ('connectivity', 'topological', 'statistical')
            
        Returns:
            Feature matrix
        """
        if feature_type == 'connectivity':
            return self.preprocess_connectome_data()
        
        elif feature_type == 'topological':
            adj_matrix = self.create_adjacency_matrix()
            
            # Calculate topological features
            features = []
            
            for i in range(adj_matrix.shape[0]):
                # Degree (number of connections)
                degree = np.sum(adj_matrix[i, :] > 0)
                
                # Clustering coefficient
                neighbors = np.where(adj_matrix[i, :] > 0)[0]
                if len(neighbors) > 1:
                    neighbor_connections = 0
                    for j in range(len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            if adj_matrix[neighbors[j], neighbors[k]] > 0:
                                neighbor_connections += 1
                    clustering = 2 * neighbor_connections / (len(neighbors) * (len(neighbors) - 1))
                else:
                    clustering = 0
                
                # Betweenness centrality (simplified)
                betweenness = np.sum(adj_matrix[i, :]) / np.sum(adj_matrix)
                
                features.append([degree, clustering, betweenness])
            
            return np.array(features)
        
        elif feature_type == 'statistical':
            adj_matrix = self.create_adjacency_matrix()
            
            # Calculate statistical features
            features = []
            
            for i in range(adj_matrix.shape[0]):
                row = adj_matrix[i, :]
                
                # Mean connection strength
                mean_strength = np.mean(row[row > 0]) if np.any(row > 0) else 0
                
                # Standard deviation of connection strength
                std_strength = np.std(row[row > 0]) if np.any(row > 0) else 0
                
                # Maximum connection strength
                max_strength = np.max(row) if np.any(row > 0) else 0
                
                # Number of strong connections (> mean + std)
                strong_connections = np.sum(row > (mean_strength + std_strength))
                
                features.append([mean_strength, std_strength, max_strength, strong_connections])
            
            return np.array(features)
        
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def create_training_data(self, feature_type: str = 'connectivity', 
                           target_type: str = 'autoencoder', 
                           test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training and testing data for neural network.
        
        Args:
            feature_type: Type of features to extract
            target_type: Type of target ('autoencoder', 'classification', 'regression')
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Extract features
        X = self.extract_features(feature_type)
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Split data
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        
        # Create targets based on target_type
        if target_type == 'autoencoder':
            # For autoencoder, targets are the same as inputs
            y_train = X_train
            y_test = X_test
        
        elif target_type == 'classification':
            # For classification, create synthetic labels based on connectivity patterns
            # This is a simplified approach - in practice, you'd use real labels
            y_train = self._create_synthetic_labels(X_train)
            y_test = self._create_synthetic_labels(X_test)
        
        elif target_type == 'regression':
            # For regression, predict some derived property
            y_train = self._create_synthetic_targets(X_train)
            y_test = self._create_synthetic_targets(X_test)
        
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        # Transpose for neural network input (features x samples)
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T
        
        return X_train, X_test, y_train, y_test
    
    def _create_synthetic_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Create synthetic classification labels based on connectivity patterns.
        
        Args:
            X: Feature matrix
            
        Returns:
            One-hot encoded labels
        """
        # Simple clustering-based labels
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Convert to one-hot encoding
        n_samples = len(labels)
        n_classes = 3
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1
        
        return one_hot
    
    def _create_synthetic_targets(self, X: np.ndarray) -> np.ndarray:
        """
        Create synthetic regression targets based on connectivity patterns.
        
        Args:
            X: Feature matrix
            
        Returns:
            Target values
        """
        # Create targets as a function of the features
        targets = np.sum(X, axis=1) + 0.1 * np.random.randn(X.shape[0])
        return targets.reshape(-1, 1)
    
    def visualize_connectome(self, save_path: Optional[str] = None):
        """
        Visualize the connectome data.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.connectome_data is None:
            print("No connectome data loaded")
            return
        
        # Create adjacency matrix
        adj_matrix = self.create_adjacency_matrix()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap
        im1 = ax1.imshow(adj_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Connectome Adjacency Matrix')
        ax1.set_xlabel('Neuron Index')
        ax1.set_ylabel('Neuron Index')
        plt.colorbar(im1, ax=ax1)
        
        # Degree distribution
        degrees = np.sum(adj_matrix > 0, axis=1)
        ax2.hist(degrees, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_title('Degree Distribution')
        ax2.set_xlabel('Number of Connections')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary containing data summary
        """
        summary = {}
        
        if self.connectome_data is not None:
            summary['connectome'] = {
                'shape': self.connectome_data.shape,
                'columns': list(self.connectome_data.columns),
                'dtypes': self.connectome_data.dtypes.value_counts().to_dict()
            }
        
        if self.neuron_types is not None:
            summary['neuron_types'] = {
                'shape': self.neuron_types.shape,
                'columns': list(self.neuron_types.columns)
            }
        
        if self.neuron_relatedness is not None:
            summary['neuron_relatedness'] = {
                'shape': self.neuron_relatedness.shape,
                'columns': list(self.neuron_relatedness.columns)
            }
        
        return summary 