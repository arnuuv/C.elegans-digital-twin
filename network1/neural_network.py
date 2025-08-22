import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable
import json
import os

class NeuralNetwork:
    """
    A feedforward neural network implementation for C. elegans connectome analysis.
    """
    
    def __init__(self, layer_sizes: List[int], activation_function: str = 'relu', 
                 learning_rate: float = 0.01, random_seed: int = 42):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            activation_function: Activation function to use ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for gradient descent
            random_seed: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for better training
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        # Set activation function
        self.activation_function = activation_function
        self.activation = self._get_activation_function()
        self.activation_derivative = self._get_activation_derivative()
        
        # Training history
        self.training_loss = []
        self.validation_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []
    
    def _get_activation_function(self) -> Callable:
        """Get the activation function."""
        if self.activation_function == 'relu':
            return lambda x: np.maximum(0, x)
        elif self.activation_function == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_function == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
    
    def _get_activation_derivative(self) -> Callable:
        """Get the derivative of the activation function."""
        if self.activation_function == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif self.activation_function == 'sigmoid':
            return lambda x: self.activation(x) * (1 - self.activation(x))
        elif self.activation_function == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (input_features, batch_size)
            
        Returns:
            Tuple of (activations, z_values) for each layer
        """
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - no activation for regression, softmax for classification
                activations.append(z)
            else:
                activations.append(self.activation(z))
        
        return activations, z_values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        activations, _ = self.forward(X)
        return activations[-1]
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, loss_type: str = 'mse') -> float:
        """
        Compute the loss between predictions and true values.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            loss_type: Type of loss function ('mse', 'cross_entropy')
            
        Returns:
            Loss value
        """
        if loss_type == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif loss_type == 'cross_entropy':
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                z_values: List[np.ndarray], loss_type: str = 'mse') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass to compute gradients.
        
        Args:
            X: Input data
            y: True labels
            activations: Activations from forward pass
            z_values: Z values from forward pass
            loss_type: Type of loss function
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = X.shape[1]
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error
        if loss_type == 'mse':
            delta = activations[-1] - y
        elif loss_type == 'cross_entropy':
            delta = activations[-1] - y
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
        
        # Backpropagate error
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for current layer
            weight_gradients[i] = np.dot(delta, activations[i].T) / m
            bias_gradients[i] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Compute error for next layer (if not input layer)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.activation_derivative(z_values[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], bias_gradients: List[np.ndarray]):
        """
        Update network parameters using computed gradients.
        
        Args:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, loss_type: str = 'mse',
              verbose: bool = True) -> dict:
        """
        Train the neural network.
        
        Args:
            X_train: Training input data
            y_train: Training labels
            X_val: Validation input data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            loss_type: Type of loss function
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        n_samples = X_train.shape[1]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward pass
                activations, z_values = self.forward(X_batch)
                
                # Compute loss
                batch_loss = self.compute_loss(activations[-1], y_batch, loss_type)
                epoch_loss += batch_loss
                
                # Compute accuracy (for classification)
                if loss_type == 'cross_entropy':
                    predictions = np.argmax(activations[-1], axis=0)
                    true_labels = np.argmax(y_batch, axis=0)
                    batch_accuracy = np.mean(predictions == true_labels)
                    epoch_accuracy += batch_accuracy
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations, z_values, loss_type)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Compute average loss and accuracy for the epoch
            epoch_loss /= n_batches
            epoch_accuracy /= n_batches
            
            # Validation
            val_loss = None
            val_accuracy = None
            
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward(X_val)
                val_loss = self.compute_loss(val_activations[-1], y_val, loss_type)
                
                if loss_type == 'cross_entropy':
                    val_predictions = np.argmax(val_activations[-1], axis=0)
                    val_true_labels = np.argmax(y_val, axis=0)
                    val_accuracy = np.mean(val_predictions == val_true_labels)
            
            # Store training history
            self.training_loss.append(epoch_loss)
            self.validation_loss.append(val_loss)
            self.training_accuracy.append(epoch_accuracy)
            self.validation_accuracy.append(val_accuracy)
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Training Loss: {epoch_loss:.4f}")
                if val_loss is not None:
                    print(f"  Validation Loss: {val_loss:.4f}")
                if epoch_accuracy is not None:
                    print(f"  Training Accuracy: {epoch_accuracy:.4f}")
                if val_accuracy is not None:
                    print(f"  Validation Accuracy: {val_accuracy:.4f}")
                print()
        
        return {
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Training and validation loss
        ax1.plot(self.training_loss, label='Training Loss')
        if self.validation_loss[0] is not None:
            ax1.plot(self.validation_loss, label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Training and validation accuracy
        if self.training_accuracy[0] is not None:
            ax2.plot(self.training_accuracy, label='Training Accuracy')
            if self.validation_accuracy[0] is not None:
                ax2.plot(self.validation_accuracy, label='Validation Accuracy')
            ax2.set_title('Accuracy Over Time')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        # Weight distributions
        for i, weight in enumerate(self.weights):
            ax3.hist(weight.flatten(), alpha=0.7, label=f'Layer {i+1}', bins=30)
        ax3.set_title('Weight Distributions')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True)
        
        # Network architecture visualization
        layer_names = [f'Input\n({self.layer_sizes[0]})']
        for i in range(1, len(self.layer_sizes) - 1):
            layer_names.append(f'Hidden {i}\n({self.layer_sizes[i]})')
        layer_names.append(f'Output\n({self.layer_sizes[-1]})')
        
        y_pos = np.arange(len(layer_names))
        ax4.barh(y_pos, self.layer_sizes)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(layer_names)
        ax4.set_xlabel('Number of Neurons')
        ax4.set_title('Network Architecture')
        ax4.grid(True, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'layer_sizes': self.layer_sizes,
            'activation_function': self.activation_function,
            'learning_rate': self.learning_rate,
            'random_seed': self.random_seed,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'training_history': {
                'training_loss': self.training_loss,
                'validation_loss': self.validation_loss,
                'training_accuracy': self.training_accuracy,
                'validation_accuracy': self.validation_accuracy
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'NeuralNetwork':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded neural network
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create network instance
        network = cls(
            layer_sizes=model_data['layer_sizes'],
            activation_function=model_data['activation_function'],
            learning_rate=model_data['learning_rate'],
            random_seed=model_data['random_seed']
        )
        
        # Load weights and biases
        network.weights = [np.array(w) for w in model_data['weights']]
        network.biases = [np.array(b) for b in model_data['biases']]
        
        # Load training history
        network.training_loss = model_data['training_history']['training_loss']
        network.validation_loss = model_data['training_history']['validation_loss']
        network.training_accuracy = model_data['training_history']['training_accuracy']
        network.validation_accuracy = model_data['training_history']['validation_accuracy']
        
        return network 