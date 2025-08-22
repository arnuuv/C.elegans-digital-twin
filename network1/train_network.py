#!/usr/bin/env python3
"""
Training script for C. elegans neural network.
This script demonstrates how to train the neural network on connectome data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from neural_network import NeuralNetwork
from data_loader import C_elegansDataLoader

def main():
    """
    Main training function.
    """
    print("C. elegans Neural Network Training")
    print("=" * 40)
    
    # Initialize data loader
    print("\n1. Loading C. elegans data...")
    data_loader = C_elegansDataLoader()
    
    try:
        # Try to load connectome data
        connectome_data = data_loader.load_connectome_data()
        if connectome_data is not None:
            print("✓ Connectome data loaded successfully")
        else:
            print("✗ Failed to load connectome data")
            return
        
        # Load additional data if available
        neuron_types = data_loader.load_neuron_types()
        neuron_relatedness = data_loader.load_neuron_relatedness()
        
        # Get data summary
        summary = data_loader.get_data_summary()
        print(f"\nData Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value['shape']}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data for demonstration
        n_neurons = 100
        n_features = 50
        
        # Generate random connectivity matrix
        np.random.seed(42)
        connectivity = np.random.randn(n_neurons, n_features)
        connectivity = (connectivity > 0.5).astype(float)  # Binary connectivity
        
        # Store in data loader for compatibility
        data_loader.connectome_data = type('obj', (object,), {
            'select_dtypes': lambda self, **kwargs: type('obj', (object,), {
                'values': lambda self: connectivity
            })()
        })()
        
        print(f"✓ Synthetic data created: {connectivity.shape}")
    
    # Create training data
    print("\n2. Preparing training data...")
    try:
        X_train, X_test, y_train, y_test = data_loader.create_training_data(
            feature_type='connectivity',
            target_type='autoencoder',
            test_size=0.2
        )
        print(f"✓ Training data prepared:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        print("Creating synthetic training data...")
        
        # Create synthetic training data
        n_features = 50
        n_samples = 100
        
        X_train = np.random.randn(n_features, n_samples)
        X_test = np.random.randn(n_features, int(n_samples * 0.2))
        y_train = X_train.copy()  # Autoencoder
        y_test = X_test.copy()
        
        print(f"✓ Synthetic training data created")
    
    # Initialize neural network
    print("\n3. Initializing neural network...")
    
    # Determine network architecture based on data
    input_size = X_train.shape[0]
    hidden_size = max(32, input_size // 2)
    output_size = y_train.shape[0]
    
    layer_sizes = [input_size, hidden_size, hidden_size // 2, output_size]
    
    print(f"Network architecture: {layer_sizes}")
    
    # Create network
    network = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation_function='relu',
        learning_rate=0.01,
        random_seed=42
    )
    
    print("✓ Neural network initialized")
    
    # Train the network
    print("\n4. Training neural network...")
    try:
        training_history = network.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=100,
            batch_size=16,
            loss_type='mse',
            verbose=True
        )
        print("✓ Training completed successfully")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Training with reduced complexity...")
        
        # Try with simpler network
        simple_layer_sizes = [input_size, 16, output_size]
        simple_network = NeuralNetwork(
            layer_sizes=simple_layer_sizes,
            activation_function='relu',
            learning_rate=0.01,
            random_seed=42
        )
        
        training_history = simple_network.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=50,
            batch_size=16,
            loss_type='mse',
            verbose=True
        )
        
        network = simple_network
        print("✓ Training completed with simplified network")
    
    # Evaluate the network
    print("\n5. Evaluating network performance...")
    
    # Training predictions
    train_predictions = network.predict(X_train)
    train_loss = network.compute_loss(train_predictions, y_train, 'mse')
    
    # Test predictions
    test_predictions = network.predict(X_test)
    test_loss = network.compute_loss(test_predictions, y_test, 'mse')
    
    print(f"Training Loss: {train_loss:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    
    # Calculate reconstruction error for autoencoder
    if X_train.shape == y_train.shape:
        train_reconstruction_error = np.mean((train_predictions - y_train) ** 2)
        test_reconstruction_error = np.mean((test_predictions - y_test) ** 2)
        
        print(f"Training Reconstruction Error: {train_reconstruction_error:.6f}")
        print(f"Test Reconstruction Error: {test_reconstruction_error:.6f}")
    
    # Visualize results
    print("\n6. Generating visualizations...")
    
    try:
        # Plot training history
        network.plot_training_history()
        
        # Plot some predictions vs actual
        if X_train.shape[0] <= 10:  # Only if we have few features
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i in range(min(4, X_train.shape[0])):
                axes[i].plot(y_train[i, :50], label='Actual', alpha=0.7)
                axes[i].plot(train_predictions[i, :50], label='Predicted', alpha=0.7)
                axes[i].set_title(f'Feature {i+1} - Training')
                axes[i].set_xlabel('Sample')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Save the trained model
    print("\n7. Saving trained model...")
    try:
        model_path = "trained_c_elegans_network.json"
        network.save_model(model_path)
        print(f"✓ Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Demonstrate model loading
    print("\n8. Demonstrating model loading...")
    try:
        loaded_network = NeuralNetwork.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Verify it's the same
        test_pred_original = network.predict(X_test[:, :5])
        test_pred_loaded = loaded_network.predict(X_test[:, :5])
        
        if np.allclose(test_pred_original, test_pred_loaded):
            print("✓ Loaded model produces identical predictions")
        else:
            print("✗ Loaded model produces different predictions")
            
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("\n" + "=" * 40)
    print("Training completed successfully!")
    print("\nNext steps:")
    print("1. Analyze the training history plots")
    print("2. Experiment with different network architectures")
    print("3. Try different feature extraction methods")
    print("4. Apply the trained network to new C. elegans data")
    print("5. Explore the saved model file")

if __name__ == "__main__":
    main() 