#!/usr/bin/env python3
"""
Simple test script to verify the neural network components work correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_neural_network():
    """Test the basic neural network functionality."""
    print("Testing Neural Network...")
    
    try:
        from neural_network import NeuralNetwork
        
        # Create a simple network
        network = NeuralNetwork(
            layer_sizes=[10, 5, 10],
            activation_function='relu',
            learning_rate=0.01
        )
        
        # Test forward pass
        X = np.random.randn(10, 20)
        activations, z_values = network.forward(X)
        
        assert len(activations) == 3, "Should have 3 activation layers"
        assert activations[0].shape == (10, 20), "Input layer shape incorrect"
        assert activations[-1].shape == (10, 20), "Output layer shape incorrect"
        
        print("✓ Forward pass works correctly")
        
        # Test prediction
        predictions = network.predict(X)
        assert predictions.shape == (10, 20), "Prediction shape incorrect"
        
        print("✓ Prediction works correctly")
        
        # Test loss computation
        loss = network.compute_loss(predictions, X, 'mse')
        assert isinstance(loss, float), "Loss should be a float"
        assert loss >= 0, "Loss should be non-negative"
        
        print("✓ Loss computation works correctly")
        
        # Test training (just a few epochs)
        y = X.copy()  # Autoencoder
        history = network.train(
            X_train=X,
            y_train=y,
            epochs=2,
            batch_size=10,
            verbose=False
        )
        
        assert 'training_loss' in history, "Training history should contain loss"
        assert len(history['training_loss']) == 2, "Should have 2 training epochs"
        
        print("✓ Training works correctly")
        
        # Test model saving/loading
        network.save_model("test_model.json")
        
        loaded_network = NeuralNetwork.load_model("test_model.json")
        
        # Verify predictions are the same
        test_pred_original = network.predict(X[:, :5])
        test_pred_loaded = loaded_network.predict(X[:, :5])
        
        assert np.allclose(test_pred_original, test_pred_loaded), "Loaded model should produce same predictions"
        
        print("✓ Model saving/loading works correctly")
        
        # Clean up
        os.remove("test_model.json")
        
        print("✓ All neural network tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_data_loader():
    """Test the data loader functionality."""
    print("\nTesting Data Loader...")
    
    try:
        from data_loader import C_elegansDataLoader
        
        # Create data loader
        data_loader = C_elegansDataLoader()
        
        # Test synthetic data creation
        n_neurons = 50
        n_features = 25
        
        # Create mock connectome data
        connectivity = np.random.randn(n_neurons, n_features)
        connectivity = (connectivity > 0.5).astype(float)
        
        # Mock the data loader's connectome data
        data_loader.connectome_data = type('obj', (object,), {
            'select_dtypes': lambda self, **kwargs: type('obj', (object,), {
                'values': lambda self: connectivity
            })()
        })()
        
        # Test feature extraction
        features = data_loader.extract_features('connectivity')
        assert features.shape == (n_neurons, n_features), "Feature extraction shape incorrect"
        
        print("✓ Feature extraction works correctly")
        
        # Test training data creation
        X_train, X_test, y_train, y_test = data_loader.create_training_data(
            feature_type='connectivity',
            target_type='autoencoder',
            test_size=0.2
        )
        
        assert X_train.shape[1] + X_test.shape[1] == n_neurons, "Train/test split incorrect"
        assert X_train.shape[0] == y_train.shape[0], "Feature/target mismatch"
        
        print("✓ Training data creation works correctly")
        
        print("✓ All data loader tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting Integration...")
    
    try:
        from neural_network import NeuralNetwork
        from data_loader import C_elegansDataLoader
        
        # Create synthetic data
        n_neurons = 30
        n_features = 20
        
        connectivity = np.random.randn(n_neurons, n_features)
        connectivity = (connectivity > 0.5).astype(float)
        
        # Mock data loader
        data_loader = C_elegansDataLoader()
        data_loader.connectome_data = type('obj', (object,), {
            'select_dtypes': lambda self, **kwargs: type('obj', (object,), {
                'values': lambda self: connectivity
            })()
        })()
        
        # Create training data
        X_train, X_test, y_train, y_test = data_loader.create_training_data(
            feature_type='connectivity',
            target_type='autoencoder',
            test_size=0.3
        )
        
        # Create and train network
        network = NeuralNetwork(
            layer_sizes=[X_train.shape[0], 16, y_train.shape[0]],
            activation_function='relu',
            learning_rate=0.01
        )
        
        # Quick training
        history = network.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=3,
            batch_size=8,
            verbose=False
        )
        
        # Test predictions
        predictions = network.predict(X_test)
        assert predictions.shape == y_test.shape, "Prediction shape mismatch"
        
        # Test loss
        loss = network.compute_loss(predictions, y_test, 'mse')
        assert isinstance(loss, float) and loss >= 0, "Invalid loss value"
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running C. elegans Neural Network Tests")
    print("=" * 40)
    
    tests = [
        test_neural_network,
        test_data_loader,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The neural network is working correctly.")
        print("\nYou can now run:")
        print("  python train_network.py     # Full training demonstration")
        print("  python example_usage.py     # Various usage examples")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main() 