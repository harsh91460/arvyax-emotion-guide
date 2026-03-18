# Edge Deployment Plan

##  Model Size
RandomForest:~ 250KB
LinearReg: ~15KB
TF-IDF Vocab: ~50KB
Total: <350KB Mobile ready

## Latency
Preprocessing: 80ms
State pred: 12ms
Intensity: 3ms
Total: <100ms  Real-time

## Mobile Strategy
1. **ONNX Export** → TensorFlow Lite
2. **Joblib → Pickle** → Binary assets  
3. **WebAssembly** for browser fallback
4. **Incremental updates** via app updates

##  Optimizations
Already lightweight (no Deep Learning)
Tree-based (fast inference)
TF-IDF precomputed vocabulary
Numerical features normalized
No external dependencies

##  Offline Robustness
- **Zero network calls**
- **Missing data imputation** built-in
- **Vocab stored locally** 
- **Model versioning** in binary
