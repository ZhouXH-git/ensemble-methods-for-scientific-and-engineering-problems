# Data-driven turbulence modeling for ASJ case

Follow the steps below to calibrate the turbulence models for ASJ using sparse observation data.

### Pre-training
Pre-training aims to initialize the weights of neural networks such that the neural-network-based turbulence model performs almost the same as the standard k-omega model.
```bash
./inputs/pretrain/run_pretrain
```

### Calibration with DAFI
```bash
./run_dafi
```
