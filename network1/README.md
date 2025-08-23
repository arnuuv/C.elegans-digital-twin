# C. elegans Neural Network

This project implements a preliminary neural network framework specifically designed for analyzing C. elegans connectome data. The network can be used for various tasks including autoencoding, classification, and regression on neural connectivity patterns.

## Project Structure

```
network1/
├── neural_network.py      # Core neural network implementation
├── data_loader.py         # C. elegans data loading and preprocessing
├── train_network.py       # Training script and demonstration
├── example_usage.py       # Various usage examples and demonstrations
├── test_network.py        # Testing suite for validation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

### Neural Network (`neural_network.py`)

- **Flexible Architecture**: Configurable layer sizes and activation functions
- **Multiple Activation Functions**: ReLU, Sigmoid, and Tanh
- **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy
- **Training Features**: Mini-batch gradient descent, validation support
- **Model Persistence**: Save and load trained models
- **Visualization**: Training history plots and network analysis

### Data Loader (`data_loader.py`)

- **Connectome Data Support**: Load and process C. elegans connectivity data
- **Feature Extraction**: Multiple feature types (connectivity, topological, statistical)
- **Data Preprocessing**: Normalization, thresholding, and cleaning
- **Synthetic Data**: Fallback to synthetic data for testing
- **Visualization**: Connectome heatmaps and degree distributions

### Training Script (`train_network.py`)

- **End-to-End Training**: Complete training pipeline demonstration
- **Error Handling**: Graceful fallbacks for missing data
- **Performance Evaluation**: Training and validation metrics
- **Model Persistence**: Save and verify trained models

### Example Usage (`example_usage.py`)

- **Multiple Examples**: Autoencoder, classification, regression, and multi-task learning
- **Real-world Scenarios**: Practical applications for C. elegans analysis
- **Configuration Examples**: Different network architectures and training setups

### Testing Suite (`test_network.py`)

- **Component Testing**: Individual module validation
- **Integration Testing**: End-to-end functionality verification
- **Error Detection**: Identify issues before production use

## Installation

1. **Navigate to the project directory**:

   ```bash
   cd C.elegans-digital-twin/network1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Test the Installation

First, verify everything works correctly:

```bash
python test_network.py
```

### 2. Run Training Demo

See the neural network in action:

```bash
python train_network.py
```

### 3. Explore Examples

Learn different usage patterns:

```bash
python example_usage.py
```

## Usage

### Basic Training

```python
from neural_network import NeuralNetwork
from data_loader import C_elegansDataLoader

# Load data
data_loader = C_elegansDataLoader()
X_train, X_test, y_train, y_test = data_loader.create_training_data(
    feature_type='connectivity',
    target_type='autoencoder'
)

# Create network
network = NeuralNetwork(
    layer_sizes=[X_train.shape[0], 64, 32, y_train.shape[0]],
    activation_function='relu',
    learning_rate=0.01
)

# Train
history = network.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_test,
    y_val=y_test,
    epochs=100,
    batch_size=32
)

# Make predictions
predictions = network.predict(X_test)
```

### Data Types

The system supports several data types:

1. **Connectivity Data**: Direct connection strengths between neurons
2. **Topological Features**: Degree, clustering coefficient, betweenness centrality
3. **Statistical Features**: Mean, standard deviation, maximum connection strengths

### Target Types

1. **Autoencoder**: Reconstruct input patterns (useful for dimensionality reduction)
2. **Classification**: Predict neuron categories (requires labeled data)
3. **Regression**: Predict continuous properties (e.g., response strength)

## Configuration

### Network Architecture

- **Layer Sizes**: Automatically determined from data, or manually specified
- **Activation Functions**: ReLU (default), Sigmoid, Tanh
- **Learning Rate**: Adjustable (default: 0.01)
- **Random Seed**: Reproducible results (default: 42)

### Training Parameters

- **Epochs**: Number of training iterations (default: 100)
- **Batch Size**: Mini-batch size for gradient descent (default: 32)
- **Validation Split**: Fraction of data for validation (default: 0.2)

## Data Sources

The system is designed to work with:

- **Connectivity Data**: Excel files (.xls, .xlsx) containing neuron connection matrices
- **Neuron Types**: Classification data for different neuron categories
- **Neuron Relatedness**: Similarity measures between neurons

## Output

### Training Results

- **Loss Curves**: Training and validation loss over time
- **Accuracy Metrics**: Classification accuracy (when applicable)
- **Reconstruction Error**: For autoencoder tasks
- **Model File**: JSON format containing weights, biases, and training history

### Visualizations

- **Training History**: Loss and accuracy plots
- **Network Architecture**: Layer size visualization
- **Weight Distributions**: Histograms of learned parameters
- **Connectome Maps**: Heatmaps of connectivity patterns

## Example Applications

1. **Dimensionality Reduction**: Use autoencoder to compress high-dimensional connectome data
2. **Pattern Recognition**: Identify common connectivity motifs in neural circuits
3. **Anomaly Detection**: Find unusual connection patterns
4. **Feature Learning**: Extract meaningful representations from raw connectivity data

## Troubleshooting

### Common Issues

1. **Missing Data**: The system will create synthetic data for demonstration
2. **Memory Issues**: Reduce batch size or network complexity
3. **Training Instability**: Lower learning rate or use different activation functions
4. **Import Errors**: Ensure all dependencies are installed

### Performance Tips

1. **Data Preprocessing**: Normalize and clean data before training
2. **Architecture**: Start with simple networks and increase complexity gradually
3. **Hyperparameters**: Experiment with learning rates and batch sizes
4. **Validation**: Use validation data to prevent overfitting

## Future Enhancements

- **Advanced Architectures**: Convolutional and recurrent neural networks
- **Transfer Learning**: Pre-trained models for specific C. elegans tasks
- **Real-time Training**: Interactive training with live data
- **Integration**: Connect with OpenWorm and other C. elegans databases
- **GPU Support**: Accelerated training using CUDA

## Contributing

This is a preliminary implementation. Contributions are welcome for:

- Additional neural network architectures
- Improved data preprocessing methods
- Better visualization tools
- Performance optimizations
- Integration with external data sources

## License

This project is part of the C. elegans Digital Twin initiative and follows the same licensing terms.

## References

- White, J.G., et al. (1986). The structure of the nervous system of the nematode Caenorhabditis elegans.
- Varshney, L.R., et al. (2011). Structural properties of the Caenorhabditis elegans neuronal network.
- Cook, S.J., et al. (2019). Whole-animal connectomes of both Caenorhabditis elegans sexes.
