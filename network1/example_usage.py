#!/usr/bin/env python3
"""
Example usage script for the C. elegans neural network.
This script demonstrates various configurations and use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from data_loader import C_elegansDataLoader

def example_autoencoder():
    """Example 1: Autoencoder for dimensionality reduction."""
    print("=" * 50)
    print("Example 1: Autoencoder for Dimensionality Reduction")
    print("=" * 50)
    
    # Create synthetic connectome data
    n_neurons = 100
    n_features = 50
    
    # Generate random connectivity matrix with some structure
    np.random.seed(42)
    connectivity = np.random.randn(n_neurons, n_features)
    
    # Add some structure (correlations between features)
    for i in range(0, n_features, 5):
        connectivity[:, i:i+3] += np.random.randn(n_neurons, 3) * 0.5
    
    # Normalize
    connectivity = (connectivity - connectivity.mean()) / connectivity.std()
    
    # Split data
    n_train = int(0.8 * n_neurons)
    X_train = connectivity[:n_train].T
    X_test = connectivity[n_train:].T
    
    # Create autoencoder (input -> compressed -> output)
    input_size = X_train.shape[0]
    compressed_size = 16
    output_size = X_train.shape[0]
    
    autoencoder = NeuralNetwork(
        layer_sizes=[input_size, compressed_size, output_size],
        activation_function='relu',
        learning_rate=0.01
    )
    
    print(f"Autoencoder architecture: {autoencoder.layer_sizes}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Compression ratio: {input_size/compressed_size:.1f}x")
    
    # Train
    history = autoencoder.train(
        X_train=X_train,
        y_train=X_train,  # Target is same as input
        X_val=X_test,
        y_val=X_test,
        epochs=50,
        batch_size=16,
        verbose=True
    )
    
    # Test reconstruction
    reconstructed = autoencoder.predict(X_test)
    reconstruction_error = np.mean((reconstructed - X_test) ** 2)
    print(f"\nReconstruction error: {reconstruction_error:.6f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Original vs reconstructed
    plt.subplot(1, 3, 1)
    plt.imshow(X_test, cmap='viridis', aspect='auto')
    plt.title('Original Data')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap='viridis', aspect='auto')
    plt.title('Reconstructed Data')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(reconstructed - X_test), cmap='hot', aspect='auto')
    plt.title('Reconstruction Error')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    
    plt.tight_layout()
    plt.show()
    
    return autoencoder

def example_classification():
    """Example 2: Classification of neuron types."""
    print("\n" + "=" * 50)
    print("Example 2: Classification of Neuron Types")
    print("=" * 50)
    
    # Create synthetic neuron classification data
    n_neurons = 200
    n_features = 30
    
    # Generate features for different neuron types
    np.random.seed(42)
    
    # Type 0: Sensory neurons (high input connectivity)
    sensory_features = np.random.randn(n_neurons//3, n_features)
    sensory_features[:, :10] += 2  # High input connectivity
    
    # Type 1: Motor neurons (high output connectivity)
    motor_features = np.random.randn(n_neurons//3, n_features)
    motor_features[:, 10:20] += 2  # High output connectivity
    
    # Type 2: Interneurons (balanced connectivity)
    interneuron_features = np.random.randn(n_neurons//3, n_features)
    interneuron_features[:, 5:15] += 1  # Moderate connectivity
    
    # Combine all types
    X = np.vstack([sensory_features, motor_features, interneuron_features])
    
    # Create labels (one-hot encoding)
    y = np.zeros((n_neurons, 3))
    y[:n_neurons//3, 0] = 1  # Sensory
    y[n_neurons//3:2*n_neurons//3, 1] = 1  # Motor
    y[2*n_neurons//3:, 2] = 1  # Interneuron
    
    # Shuffle data
    indices = np.random.permutation(n_neurons)
    X = X[indices]
    y = y[indices]
    
    # Split data
    n_train = int(0.8 * n_neurons)
    X_train = X[:n_train].T
    X_test = X[n_train:].T
    y_train = y[:n_train].T
    y_test = y[n_train:].T
    
    # Create classification network
    classifier = NeuralNetwork(
        layer_sizes=[X_train.shape[0], 64, 32, y_train.shape[0]],
        activation_function='relu',
        learning_rate=0.01
    )
    
    print(f"Classifier architecture: {classifier.layer_sizes}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {y_train.shape[0]}")
    
    # Train
    history = classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        batch_size=32,
        loss_type='cross_entropy',
        verbose=True
    )
    
    # Test classification
    predictions = classifier.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=0)
    true_classes = np.argmax(y_test, axis=0)
    
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"\nClassification accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Sensory', 'Motor', 'Interneuron']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    return classifier

def example_regression():
    """Example 3: Regression to predict neuron response strength."""
    print("\n" + "=" * 50)
    print("Example 3: Regression to Predict Neuron Response Strength")
    print("=" * 50)
    
    # Create synthetic regression data
    n_neurons = 150
    n_features = 25
    
    # Generate features (connectivity patterns)
    np.random.seed(42)
    X = np.random.randn(n_neurons, n_features)
    
    # Create target (response strength) as a function of features
    # Response strength depends on input connectivity and some nonlinear interactions
    response_strength = (
        0.3 * np.sum(X[:, :10], axis=1) +  # Input connectivity
        0.2 * np.sum(X[:, 10:20], axis=1) +  # Output connectivity
        0.1 * np.sum(X[:, 20:], axis=1) +  # Other features
        0.1 * np.sum(X[:, :10] * X[:, 10:20], axis=1) +  # Interaction term
        np.random.randn(n_neurons) * 0.1  # Noise
    )
    
    # Normalize target
    response_strength = (response_strength - response_strength.mean()) / response_strength.std()
    
    # Split data
    n_train = int(0.8 * n_neurons)
    X_train = X[:n_train].T
    X_test = X[n_train:].T
    y_train = response_strength[:n_train].reshape(1, -1)
    y_test = response_strength[n_train:].reshape(1, -1)
    
    # Create regression network
    regressor = NeuralNetwork(
        layer_sizes=[X_train.shape[0], 32, 16, y_train.shape[0]],
        activation_function='relu',
        learning_rate=0.01
    )
    
    print(f"Regressor architecture: {regressor.layer_sizes}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Train
    history = regressor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=80,
        batch_size=24,
        loss_type='mse',
        verbose=True
    )
    
    # Test regression
    predictions = regressor.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    r_squared = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    
    print(f"\nMean Squared Error: {mse:.6f}")
    print(f"R-squared: {r_squared:.3f}")
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Predictions vs actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test.flatten(), predictions.flatten(), alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Response Strength')
    plt.ylabel('Predicted Response Strength')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    residuals = predictions.flatten() - y_test.flatten()
    plt.subplot(1, 3, 2)
    plt.scatter(predictions.flatten(), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Response Strength')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Feature importance (simplified)
    plt.subplot(1, 3, 3)
    feature_importance = np.abs(regressor.weights[0]).mean(axis=0)
    top_features = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(top_features)), feature_importance[top_features])
    plt.yticks(range(len(top_features)), [f'F{i+1}' for i in top_features])
    plt.xlabel('Average Weight Magnitude')
    plt.title('Top 10 Feature Importances')
    
    plt.tight_layout()
    plt.show()
    
    return regressor

def example_advanced_usage():
    """Example 4: Advanced usage with custom configurations."""
    print("\n" + "=" * 50)
    print("Example 4: Advanced Usage with Custom Configurations")
    print("=" * 50)
    
    # Create a more complex dataset
    n_neurons = 300
    n_features = 40
    
    np.random.seed(42)
    X = np.random.randn(n_neurons, n_features)
    
    # Add some complex patterns
    # Pattern 1: Clusters of neurons with similar connectivity
    cluster_size = 50
    for i in range(0, n_neurons, cluster_size):
        cluster_features = np.random.randn(cluster_size, 10)
        X[i:i+cluster_size, :10] += cluster_features * 0.5
    
    # Pattern 2: Hierarchical structure
    X[:n_neurons//2, 20:30] += np.random.randn(n_neurons//2, 10) * 0.3
    X[n_neurons//2:, 30:40] += np.random.randn(n_neurons//2, 10) * 0.3
    
    # Normalize
    X = (X - X.mean()) / X.std()
    
    # Create multiple targets for multi-task learning
    # Target 1: Response strength
    target1 = np.sum(X[:, :20], axis=1) + np.random.randn(n_neurons) * 0.1
    
    # Target 2: Classification (3 classes)
    target2 = np.zeros((n_neurons, 3))
    target2[:n_neurons//3, 0] = 1
    target2[n_neurons//3:2*n_neurons//3, 1] = 1
    target2[2*n_neurons//3:, 2] = 1
    
    # Split data
    n_train = int(0.8 * n_neurons)
    X_train = X[:n_train].T
    X_test = X[n_train:].T
    
    y1_train = target1[:n_train].reshape(1, -1)
    y1_test = target1[n_train:].reshape(1, -1)
    y2_train = target2[:n_train].T
    y2_test = target2[n_train:].T
    
    # Create multi-task network
    # Shared layers + task-specific heads
    shared_layers = [X_train.shape[0], 64, 32]
    
    # Network 1: Regression
    regressor = NeuralNetwork(
        layer_sizes=shared_layers + [y1_train.shape[0]],
        activation_function='relu',
        learning_rate=0.005
    )
    
    # Network 2: Classification
    classifier = NeuralNetwork(
        layer_sizes=shared_layers + [y2_train.shape[0]],
        activation_function='relu',
        learning_rate=0.005
    )
    
    print("Multi-task learning setup:")
    print(f"  Shared layers: {shared_layers}")
    print(f"  Regression head: {shared_layers[-1]} -> {y1_train.shape[0]}")
    print(f"  Classification head: {shared_layers[-1]} -> {y2_train.shape[0]}")
    
    # Train both networks
    print("\nTraining regression network...")
    regressor.train(
        X_train=X_train,
        y_train=y1_train,
        X_val=X_test,
        y_val=y1_test,
        epochs=60,
        batch_size=32,
        verbose=False
    )
    
    print("Training classification network...")
    classifier.train(
        X_train=X_train,
        y_train=y2_train,
        X_val=X_test,
        y_val=y2_test,
        epochs=60,
        batch_size=32,
        verbose=False
    )
    
    # Evaluate both tasks
    reg_predictions = regressor.predict(X_test)
    reg_mse = np.mean((reg_predictions - y1_test) ** 2)
    
    class_predictions = classifier.predict(X_test)
    class_accuracy = np.mean(np.argmax(class_predictions, axis=0) == np.argmax(y2_test, axis=0))
    
    print(f"\nMulti-task Results:")
    print(f"  Regression MSE: {reg_mse:.6f}")
    print(f"  Classification Accuracy: {class_accuracy:.3f}")
    
    # Compare architectures
    plt.figure(figsize=(15, 5))
    
    # Training history comparison
    plt.subplot(1, 3, 1)
    plt.plot(regressor.training_loss, label='Regression', alpha=0.8)
    plt.plot(classifier.training_loss, label='Classification', alpha=0.8)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Weight distribution comparison
    plt.subplot(1, 3, 2)
    plt.hist(regressor.weights[0].flatten(), alpha=0.7, label='Regression', bins=30)
    plt.hist(classifier.weights[0].flatten(), alpha=0.7, label='Classification', bins=30)
    plt.title('First Layer Weight Distributions')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance comparison
    plt.subplot(1, 3, 3)
    reg_importance = np.abs(regressor.weights[0]).mean(axis=0)
    class_importance = np.abs(classifier.weights[0]).mean(axis=0)
    
    plt.scatter(reg_importance, class_importance, alpha=0.6)
    plt.plot([0, max(reg_importance.max(), class_importance.max())], 
             [0, max(reg_importance.max(), class_importance.max())], 'r--')
    plt.xlabel('Regression Feature Importance')
    plt.ylabel('Classification Feature Importance')
    plt.title('Feature Importance Correlation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return regressor, classifier

def main():
    """Run all examples."""
    print("C. elegans Neural Network - Example Usage")
    print("This script demonstrates various neural network configurations")
    print("and applications for C. elegans connectome analysis.\n")
    
    try:
        # Run examples
        autoencoder = example_autoencoder()
        classifier = example_classification()
        regressor = example_regression()
        multi_reg, multi_class = example_advanced_usage()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)
        
        # Save some models for later use
        print("\nSaving example models...")
        autoencoder.save_model("example_autoencoder.json")
        classifier.save_model("example_classifier.json")
        regressor.save_model("example_regressor.json")
        print("Models saved successfully!")
        
        print("\nYou can now:")
        print("1. Load these models using NeuralNetwork.load_model()")
        print("2. Apply them to new C. elegans data")
        print("3. Use them as starting points for transfer learning")
        print("4. Analyze the learned representations")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 