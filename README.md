# Financial Fraud Detection GNN

This project implements a Graph Neural Network (GNN) for detecting financial fraud using transaction data.

## Overview

The project uses PyTorch and PyTorch Geometric to build and train a GNN model on transaction data. The goal is to classify transactions as fraudulent or non-fraudulent.

## Features

- Graph-based representation of transaction data
- GNN model for classification
- Evaluation metrics including accuracy, precision, recall, and F1-score
- Visualization of training metrics and ROC curve

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mmannan17/Financial_Fraud_DetectionGNN.git
   cd Financial_Fraud_DetectionGNN
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data files:
   - `elliptic_txs_features.csv`: Node features
   - `elliptic_txs_edgelist.csv`: Edge list
   - `elliptic_txs_classes.csv`: Class labels

2. Run the training script:
   ```bash
   python ff_gnn.py
   ```

3. View the results:
   - Confusion matrix: `confusion_matrix.png`
   - ROC curve: `roc_curve.png`
   - Training metrics: `training_metrics.png`

## Results

During training, the model achieved the following performance metrics:

- **Test Accuracy**: 97.47%
- **Test Precision**: 97.47%
- **Test Recall**: 100.00%
- **Test F1 Score**: 98.72%

### Confusion Matrix

| Predicted Negative | Predicted Positive |
|--------------------|--------------------|
| 0                  | 169                |
| 0                  | 6518               |

### Training Progress

- **Epoch 01**: Loss: 2003620.6250, Val Loss: 668804.1875, Val Acc: 90.84%
- **Epoch 02**: Loss: 2680480.7500, Val Loss: 859666.5625, Val Acc: 90.84%
- **Epoch 03**: Loss: 3469554.5000, Val Loss: 836082.1250, Val Acc: 90.84%
- **Epoch 04**: Loss: 3422597.0000, Val Loss: 709417.6875, Val Acc: 90.84%
- **Epoch 05**: Loss: 2920227.5000, Val Loss: 541518.4375, Val Acc: 90.84%
- **Epoch 06**: Loss: 2217199.7500, Val Loss: 346993.0000, Val Acc: 90.84%

Early stopping was triggered during training.

### Graph Data

- **Edge Index Shape**: [2, 234355]
- **Max Index in Edge Index**: 203768
- **Min Index in Edge Index**: 0
- **Number of Nodes**: 203769

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact Mustafa at [your.email@example.com].
