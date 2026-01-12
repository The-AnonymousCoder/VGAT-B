# VGAT-B: Copyright Authentication Framework for Vector Geographic Data via Graph Attention Networks and Blockchain

## Overview

VGAT-B represents a comprehensive framework for copyright authentication of vector geographic data, integrating zero-watermarking techniques with blockchain-based verification mechanisms. This repository contains the complete codebase implementation, featuring a dual-stream graph attention network architecture for robust feature extraction and a hybrid off-chain/on-chain authentication system for decentralized copyright management.

## Key Features

- **Graph Attention Network Architecture**: Dual-stream encoder with node-level and graph-level attention mechanisms for capturing both local topological relationships and global geometric semantics
- **Zero-watermarking Technology**: Non-intrusive copyright protection that preserves data fidelity while ensuring robustness against composite attacks
- **Blockchain Integration**: Decentralized copyright authentication using smart contracts and distributed storage
- **Multi-objective Optimization**: Joint optimization strategy balancing discriminability and attack robustness
- **Comprehensive Evaluation**: Extensive experimental framework with baseline comparisons and ablation studies

## Repository Structure

### Core Algorithm Implementation

#### `VGAT/`
Core graph attention network implementation and training framework
- `VGAT-IMPROVED.py`: Main VGAT-B model implementation with dual-stream architecture
- `VGAT_with_comments.py`: Well-documented version of the VGAT implementation
- Ablation study implementations (`Ablation1_NodeOnly.py` through `Ablation5_GCN.py`)
- Training and evaluation utilities (`diagnose_training.py`, `verify_checkpoint.py`, etc.)
- `VGCN.py`: Graph convolutional network baseline implementation

#### `ZeroWatermark/`
Zero-watermark generation and verification algorithms
- Dataset-specific implementations for training and test sets
- Ablation study variants for different architectural components
- Core watermark embedding and extraction logic

#### `BlockChain/`
Blockchain-based copyright authentication system
- `contracts/VectorMapRegistry.sol`: Smart contract for copyright registration and verification
- `scripts/`: Web3 integration scripts for blockchain interaction
- `hardhat.config.js`: Hardhat configuration for smart contract deployment

### Data Processing Pipeline

#### `convertToGeoJson/`
Vector geographic data preprocessing utilities
- Shapefile to GeoJSON conversion (`convertToGeoJson.py`)
- Geometric validation and multipart handling (`check_shp_multiparts.py`, `force_shp_multiparts.py`)
- Test and training set processing scripts

#### `convertToGeoJson-Attacked/`
Attack simulation and data augmentation
- Noise injection and geometric transformation attacks
- Training and test set attack generation
- Multi-attack scenario simulation

#### `convertToGraph/`
Graph construction and feature engineering
- Delaunay triangulation-based graph generation
- Feature representation and normalization
- Graph quality validation (`check_abnormal_features.py`, `test_delaunay_fix.py`)

### Evaluation and Benchmarking

#### `zNC-Test/`
Normalized Correlation (NC) evaluation framework
- Comprehensive NC testing across different attack scenarios
- Ablation study evaluation scripts
- Statistical analysis and visualization (`Fig1.py` through `Fig12.py`)
- Graph-level NC uniqueness analysis

#### `zzContrastExperiment/`
Baseline algorithm implementations for comparative evaluation
- `Lin18/`: Lin et al. (2018) DWT-based watermarking
- `Tan24/`: Tan et al. (2024) statistical feature-based approach
- `Wu25/`: Wu et al. (2025) hybrid domain watermarking
- `Xi24/`: Xi et al. (2024) Arnold transform-based method
- `Xi25/`: Xi et al. (2025) SDWT-based watermarking

#### `zzGenerateAcademicGraphs/`
Academic visualization and analysis framework
- Publication-quality figure generation
- Statistical analysis and comparison plots
- Performance metric computation and visualization

### Utility Scripts

#### `scripts/`
General-purpose utilities
- `sync_pso_data.py`: Data synchronization utilities

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- NetworkX 2.8+
- GeoPandas 0.12+
- Node.js 16+ (for blockchain components)
- Hardhat 2.12+

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install networkx geopandas shapely
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
```

### Blockchain Setup
```bash
cd BlockChain
npm install
npx hardhat compile
```

## Usage

### Training VGAT-B Model
```python
from VGAT.VGAT_IMPROVED import VGAT_B

# Initialize model
model = VGAT_B(node_dim=64, graph_dim=128, hidden_dim=256)

# Training loop
# [Implementation details in VGAT-IMPROVED.py]
```

### Zero-watermark Generation
```python
from ZeroWatermark.zeroWatermark_TrainingSet import generate_watermark

# Generate zero-watermark for vector data
watermark = generate_watermark(graph_data, secret_key)
```

### Blockchain Registration
```javascript
const { registerCopyright } = require('./BlockChain/scripts/setup_web3_token.js');

// Register copyright on blockchain
await registerCopyright(vectorDataHash, watermarkFingerprint);
```

## Performance Metrics

The framework achieves state-of-the-art performance in robustness against composite attacks:
- **Normalized Correlation (NC)**: 0.94 under extreme composite attacks
- **Geometric Transformation Resilience**: Maintains integrity under rotation, scaling, translation
- **Topological Attack Robustness**: Resists vertex deletion, addition, and reorganization
- **Storage Efficiency**: Sub-linear scaling with data complexity


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions to improve the framework. Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure all tests pass

## Contact

For questions or collaboration opportunities, please contact the maintainers.