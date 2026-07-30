# AutoML Architectural Upgrade Specification: GPU Acceleration & Large Dataset Scalability

## Overview

This specification details the end-to-end upgrade roadmap for the **Autonomous AutoML Framework**. The upgrades address two critical production bottlenecks:
1. **Hardware Acceleration**: Leveraging CUDA / MPS GPUs for high-throughput PyTorch training, mixed-precision math, and VRAM management.
2. **Large Dataset Scalability**: Scaling beyond small tabular datasets to handle 100,000+ rows efficiently through streaming `DataLoader` batching, memory downcasting, stratified search sub-sampling, and chunked validation.

---

## Architecture Blueprint

```
+-----------------------------------------------------------------------------------+
|                                  USER / FRONTEND                                  |
+-----------------------------------------------------------------------------------+
                                          | CSV Upload & Config
                                          v
+-----------------------------------------------------------------------------------+
| PHASE 1: DATA PREPARATION & SCALABLE AUGMENTATION                                 |
| - Automatic Pandas/Polars Memory Downcasting (float64 -> float32, int64 -> int32) |
| - Adaptive Class Balancing: SMOTE for < 50k rows; Class Weighting for > 50k rows  |
+-----------------------------------------------------------------------------------+
                                          | Preprocessed Data + Metadata
                                          v
+-----------------------------------------------------------------------------------+
| STRATIFIED SUB-SAMPLER (For Search Efficiency)                                   |
| - If rows > 50k: Draw stratified subset (e.g. 30k rows) for Phase 2 & 3           |
+-----------------------------------------------------------------------------------+
                    | Sub-sampled Data                  | Sub-sampled Data
                    v                                   v
+------------------------------------+ +------------------------------------+
| PHASE 2: NAS (GPU Accelerated)     | | PHASE 3: HPO (Optuna + GPU)        |
| - GPU Tensor Placement             | | - Accelerated Trial Runs           |
| - AMP Mixed-Precision (FP16)       | | - VRAM Garbage Collection           |
| - Candidate Memory Cleanup         | | - Batch-level Evaluations          |
+------------------------------------+ +------------------------------------+
                    \                                   /
                     \ Best Arch + Best Hyperparameters /
                      v                            v
+-----------------------------------------------------------------------------------+
| PHASE 4: FULL DATASET TRAINING (GPU + Mixed Precision)                             |
| - Trains on 100% Full Dataset via PyTorch DataLoader                              |
| - AMP (Automatic Mixed Precision: torch.cuda.amp.autocast)                        |
| - Non-blocking GPU Transfer (pin_memory=True)                                     |
| - Chunked Validation Accumulation (Eliminates Out-Of-Memory Crashes)              |
+-----------------------------------------------------------------------------------+
                                          | Trained Model Checkpoint
                                          v
+-----------------------------------------------------------------------------------+
| PHASE 5: COMPRESSION & HARDWARE HANDSHAKE                                         |
| - GPU Knowledge Distillation (Teacher -> Student)                                 |
| - CPU Offloading Handshake (`model.to("cpu")`)                                    |
| - PyTorch Dynamic INT8 Quantization (`quantize_dynamic` on CPU)                   |
| - ONNX Export & Verification                                                      |
+-----------------------------------------------------------------------------------+
```

---

## Pillar 1: GPU Acceleration Technical Specification

### 1.1 Centralized Device & VRAM Manager (`src/utils/device.py`)

A unified utility module manages hardware placement and VRAM cleanup across all phases.

```python
import gc
import torch

def get_device() -> torch.device:
    """Selects the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clear_vram():
    """Flushes PyTorch CUDA cache and forces Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 1.2 Mixed-Precision Training (AMP)
To double training throughput and reduce VRAM usage by ~50%, all GPU training loops utilize PyTorch's native Automatic Mixed Precision (`torch.cuda.amp`):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=(device.type == "cuda"))

for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
    opt.zero_grad()
    
    with autocast(enabled=(device.type == "cuda")):
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
```

### 1.3 Phase 5 CPU Handshake Protocol
Dynamic quantization (`torch.quantization.quantize_dynamic`) is optimized specifically for CPU inference engines and requires CPU-bound parameters.

**Execution Protocol:**
1. Perform Knowledge Distillation between teacher and student models on GPU.
2. Transfer student model to CPU: `student = student.to("cpu")`.
3. Apply `quantize_dynamic` to Linear layers using `torch.qint8`.
4. Export the CPU-bound ONNX model.

---

## Pillar 2: Large Dataset Scalability Specification (>100k Rows)

### 2.1 Pandas Memory Optimization & Downcasting (`Phase 1`)
Default CSV reading in Pandas assigns 64-bit types (`float64`, `int64`), doubling memory usage.

```python
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
```

### 2.2 Adaptive Class Balancing
- **Small Datasets (< 50,000 rows)**: Use standard k-NN SMOTE interpolation ([phase1_data.py](file:///e:/automl/AutoML-backend/src/phases/phase1_data.py#L26)).
- **Large Datasets (> 50,000 rows)**: Disable synthetic sample generation to avoid exponential VRAM/RAM inflation. Instead, calculate inverse class frequencies and pass `pos_weight` / `weight` directly into loss functions:

```python
# Compute loss weights for imbalanced large datasets
class_counts = np.bincount(y_train)
class_weights = torch.tensor([sum(class_counts) / c for c in class_counts], dtype=torch.float32).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

### 2.3 Stratified Search Sub-Sampling (Phases 2 & 3)
Searching 6 architectures across 25 HPO trials on 500,000+ rows is computationally inefficient. 

```python
def get_search_subsample(X: np.ndarray, y: np.ndarray, max_samples: int = 30000):
    if len(X) <= max_samples:
        return X, y
    from sklearn.model_selection import train_test_split
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=42
    )
    return X_sub, y_sub
```
- **Phase 2 (NAS)** and **Phase 3 (HPO)** run on `X_sub`, `y_sub` (~30,000 rows).
- **Phase 4 (Full Training)** receives the 100% complete dataset (`X_train`, `y_train`).

### 2.4 Streaming PyTorch `DataLoader`
Replace raw array slices with non-blocking memory transfer:

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    num_workers=2 if os.cpu_count() > 2 else 0
)
```

### 2.5 Chunked Memory-Safe Validation Pass
Replacing single-pass validation (`val_out = model(Xv)`) with a chunked evaluation loop prevents CUDA OOM on large validation sets:

```python
def evaluate_model_batched(model, val_loader, loss_fn, device, task_type):
    model.eval()
    total_loss, correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            out = model(X_b)
            loss = loss_fn(out, y_b)
            total_loss += loss.item() * len(y_b)
            if task_type == "classification":
                preds = out.argmax(dim=1)
                correct += (preds == y_b).sum().item()
            total_samples += len(y_b)
            
    val_loss = total_loss / total_samples
    val_acc = (correct / total_samples * 100.0) if task_type == "classification" else None
    return val_loss, val_acc
```

---

## Step-by-Step Code Modification Checklist

### Step 1: Create `src/utils/device.py`
- [ ] Add `get_device()` to auto-detect CUDA/MPS/CPU.
- [ ] Add `clear_vram()` helper function.

### Step 2: Update `src/phases/phase1_data.py`
- [ ] Add float32/int32 memory downcasting for loaded Pandas DataFrames.
- [ ] Add dataset size check: if rows > 50,000, switch from SMOTE to loss weighting.

### Step 3: Update `src/phases/phase2_nas.py`
- [ ] Import `get_device` and `clear_vram`.
- [ ] Add dataset sub-sampling (`max_samples=30000`).
- [ ] Move model and tensor evaluation to `device`.
- [ ] Call `clear_vram()` after evaluating each architecture candidate.

### Step 4: Update `src/phases/phase3_hpo.py`
- [ ] Apply sub-sampling and GPU device placement during Optuna trials.
- [ ] Wrap trial training loops in `autocast()`.
- [ ] Call `clear_vram()` after trial completions.

### Step 5: Update `src/phases/phase4_train.py`
- [ ] Replace monolithic array passes with PyTorch `DataLoader` (`pin_memory=True`).
- [ ] Implement `autocast()` and `GradScaler()` for FP16 training.
- [ ] Replace `val_out = model(Xv)` with `evaluate_model_batched()`.
- [ ] Save best model weights on GPU/CPU safely (`copy.deepcopy(model.state_dict())`).

### Step 6: Update `src/phases/phase5_compress.py`
- [ ] Execute Knowledge Distillation loop on GPU with mini-batches.
- [ ] Perform CPU transfer handshake (`model.to("cpu")`).
- [ ] Run `quantize_dynamic` on CPU.
- [ ] Export ONNX model.

---

## Expected Performance Improvements

| Metric | Current CPU Baseline | Upgraded GPU + DataLoader | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Speed (100k rows)** | ~45 - 90 seconds | ~5 - 12 seconds | **~6x - 8x Faster** |
| **Peak RAM Footprint (100k rows)** | ~1.8 GB | ~450 MB | **~75% Reduction** |
| **Max Supported Dataset Size** | ~100,000 rows | 2,000,000+ rows | **20x Scale** |
| **VRAM Safety** | N/A (CPU only) | OOM Protected | **Zero CUDA OOMs** |
