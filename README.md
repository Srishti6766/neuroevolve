# NEUROEVOLVE — Self-Evolving Neural Architecture Search Engine

**NEUROEVOLVE** is a full-stack, 100% local, free and open-source Neural Architecture Search (NAS) platform powered by **PyTorch**, **FastAPI**, **React**, **Tailwind CSS**, **Recharts**, **NetworkX**, and **SQLite**.

It dynamically constructs, trains, benchmarks, and evolves real PyTorch neural network architectures across generations using **Genetic Algorithms**, a lightweight **Reinforcement Learning (Q-Learning + UCB1 Contextual Bandit)** controller, and **NSGA-II Pareto Multi-Objective Optimization**.

---

## Key Features

- **100% Real PyTorch Computation**: Every candidate network is built as a live `torch.nn.Module`, trained on your CPU, evaluated with `scikit-learn` (Accuracy, Precision, Recall, F1), benchmarked for inference latency, and saved as a real `.pt` checkpoint. Zero fabricated metrics.
- **Genetic Evolution Engine**: Population initialization, tournament selection, elitism, structural layer crossover, and 8 mutation operators (`add_layer`, `remove_layer`, `units`, `activation`, `dropout`, `batch_norm`, `optimizer`, `learning_rate`).
- **Reinforcement Learning Policy**: Lightweight Tabular Q-Learning + UCB1 Contextual Bandit that updates Q-values from real evaluated multi-objective rewards and proposes promising architectures.
- **Pareto Multi-Objective Optimization**: NSGA-II non-dominated sorting and crowding distance across **Validation Accuracy**, **Inference Latency (ms)**, and **Model Size (KB)**, with 3 modes: *Accuracy-Focused*, *Speed-Focused*, and *Balanced Multi-Objective*.
- **Real-Time WebSocket Streaming**: Live epoch-by-epoch loss/accuracy curves, candidate rankings, experiment logs, and safe **Pause / Resume / Stop** controls.
- **8 Interactive Research Pages**:
  1. **Landing Page** (`/`) — Animated neural network canvas, live telemetry strip, pipeline overview.
  2. **Main Dashboard** (`/dashboard`) — Real-time KPIs, accuracy/latency/size/fitness generation charts, and Pareto frontier scatter plot.
  3. **Experiment Configuration** (`/configure`) — Dataset selector (MNIST, Fashion-MNIST, Offline Digits, Custom CSV upload), search space designer, optimization modes, and CPU thread limits.
  4. **Live Experiment Monitor** (`/monitor`) — Live candidate topology, streaming epoch charts, candidate leaderboard, log console, Pause/Resume/Stop.
  5. **Architecture Explorer** (`/explorer`) — Interactive SVG layer visualizer (Input, Dense, BatchNorm, Activation, Dropout, Output) with parameter inspection and pinned side-by-side comparison.
  6. **Evolution History** (`/evolution`) — Interactive NetworkX genealogy DAG showing parent-to-child crossover/mutation/RL paths and learned RL Q-values.
  7. **Model Comparison** (`/compare`) — Multi-select comparison charts (Accuracy, Precision, Recall, F1, Latency, Size, Pareto) and ranked matrix.
  8. **Best Architecture** (`/best`) — Champion model showcase, training history curves, and one-click `.pt` PyTorch weights + `.json` config downloads.

---

## Project Structure

```text
neuroevolve/
├── frontend/                  # React + Vite + Tailwind + Recharts + Framer Motion
│   └── src/
│       ├── api/               # REST & WebSocket client
│       ├── components/        # Layout, NeuralCanvas, ArchitectureGraph, EvolutionGraph, ParetoChart
│       └── pages/             # All 8 pages
├── backend/                   # FastAPI + PyTorch NAS Engine
│   ├── main.py                # FastAPI entrypoint & WebSocket endpoints
│   ├── config.py              # Local paths configuration
│   ├── api/                   # REST routes, Pydantic schemas, WebSocket manager
│   ├── engine/                # SearchSpace & async NASEngineManager
│   ├── models/                # DynamicNeuralNet PyTorch builder
│   ├── evolution/             # Genetic operators & NetworkX lineage graph
│   ├── reinforcement_learning/# Tabular Q-Learning + UCB1 Bandit controller
│   ├── optimization/          # NSGA-II Pareto frontier & multi-objective fitness
│   ├── training/              # CPU PyTorch trainer with pause/stop hooks
│   ├── evaluation/            # Metrics, latency benchmarking & checkpoint saver
│   ├── database/              # SQLite persistence (WAL mode)
│   ├── datasets/              # MNIST, Fashion-MNIST, UCI Digits, Custom CSV uploader
│   └── tests/                 # End-to-end pytest verification suite
├── saved_models/              # Saved PyTorch .pt checkpoints
├── experiments/               # Exported experiment JSON summaries
├── requirements.txt
└── README.md
```

---

## Local Setup & Startup Instructions (Windows / Ryzen 5 CPU)

### Prerequisites
- Python 3.10+ (CPU PyTorch works cleanly on Windows Ryzen 5 laptops)
- Node.js 18+ and npm

### 1. Install Backend Dependencies
```powershell
cd neuroevolve
pip install -r requirements.txt
```
*(If PyTorch is not yet installed, install CPU-only PyTorch via `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.)*

### 2. Install Frontend Dependencies
```powershell
cd frontend
npm install
```

### 3. Start the Backend (Terminal 1)
From the `neuroevolve` root directory:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Backend API: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### 4. Start the Frontend (Terminal 2)
From `neuroevolve/frontend`:
```powershell
npm run dev
```
- Open the website at **`http://127.0.0.1:5173`** (or `http://localhost:5173`).

---

## CPU-Friendly Defaults (Ryzen 5 Optimization)

NEUROEVOLVE ships with fast CPU defaults so experiments complete in seconds rather than hours:
- **Default Dataset**: MNIST (with stratified CPU-friendly subset cap of `4,000` train / `1,000` val samples — configurable up to full `60,000 / 10,000`).
- **Generations**: `3` | **Population Size**: `4` | **Max Epochs**: `3` | **Batch Size**: `64`
- **CPU Threads**: Configurable via `torch.set_num_threads(...)` (default `4` threads).
- **Offline Fallback**: Includes scikit-learn's built-in **UCI Handwritten Digits (8x8)** dataset and a **Custom CSV Uploader** that work 100% offline.

---

## Measurement Methodology & Documented Limitations

- **Validation Accuracy / Precision / Recall / F1**: Computed on the held-out validation split using `scikit-learn` (`weighted` averaging).
- **Inference Latency (`ms`)**: Measured on CPU in `model.eval()` with `torch.no_grad()` after 10 warmup passes, averaging 60 single-sample forward passes via `time.perf_counter()`. Results can vary slightly with OS background load and CPU frequency scaling.
- **Model File Size (`KB`)**: Exact on-disk byte size of the saved PyTorch `.pt` checkpoint (`state_dict` + metadata).
- **Peak Memory (`MB`)**: Combines Python's `tracemalloc` peak heap allocation during training with exact PyTorch parameter, gradient, and optimizer state tensor bytes (`~4×` parameter bytes). Because Python shares a process allocator across threads, this provides a consistent architecture-comparable estimate rather than isolated hardware-level RSS.

---

## Running Automated Verification Tests

From the `neuroevolve` root directory:
```powershell
python -m pytest backend/tests/test_nas_pipeline.py -v
```

---

## Troubleshooting

- **MNIST download error on first run**: Ensure internet access for the initial `torchvision` MNIST download into `backend/datasets/data/`, or select **UCI Handwritten Digits (8x8 Offline)** or upload a CSV dataset.
- **Frontend shows "OFFLINE" in sidebar**: Confirm the FastAPI backend is running on `http://127.0.0.1:8000`. Vite proxies `/api` and `/ws` directly to port `8000`.
- **Port 8000 or 5173 already in use**: Stop conflicting processes or specify another port (`--port 8001` and update `frontend/vite.config.js`).
- **Database persistence**: All history is stored in `backend/database/neuroevolve.db`. Restarting the server safely marks any interrupted runs as `stopped` while preserving all evaluated architectures and checkpoints.
