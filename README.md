#  Sequential Modeling Experiments for Financial Market Prediction

This repository contains various experimental deep learning models designed to explore **sequential patterns and dependencies** in trading data—such as 3-hour candle structures, liquidity sweeps, and volatility setups.

## ⚙️ Core Focus
The primary goal of these models is to identify **structured behavioral patterns** in market data—particularly the early phases of manipulation setups (e.g., raids, displacements, reversals). These experiments focus on understanding **sequence-to-behavior mapping**, using various architectures and inputs.

---

##  Model Architectures

This repo contains:

- `dual_input_model.py`  
  Combines two parallel inputs into a unified prediction model.

- `dual_input_sequential.py`  
  RNN-based architecture with parallel sequence streams.

- `dual_input_sequential_focal.py`  
  Adds focal loss adjustments to handle class imbalance in rare event prediction.

- `dual_input_with_normalization.py`  
  Incorporates dataset normalization and memory-efficient preprocessing.

- `sequential_simulation.py`  
  Mock simulation of trade sessions using historical data as context memory.

- `simple_sequential.py`  
  Lightweight version to test hypotheses with reduced compute overhead.

---

##  Notes

- These are **experimental** and many are not fully production-ready.
- Most models are designed for use on **pre-processed structured datasets** extracted from earlier Wormsign backtesting logs.
- Architecture and naming conventions may evolve as deeper patterns and biases are uncovered through validation and manual chart replay.

---

##  Next Steps

- Add TensorBoard and visualization tools  
- Benchmark prediction accuracy on known historical trade setups  
- Explore hybrid models incorporating **contextual state data + sequential price flow**
