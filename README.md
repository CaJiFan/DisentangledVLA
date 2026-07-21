# DisentangledVLA

This repository contains the codebase for training Conditional Flow Matching (CFM) projectors on top of foundation Vision-Language-Action (VLA) models (like OpenVLA and Octo) using a prior-conditioned VAE bottleneck.

## Repository Setup

This repository relies on several third-party submodules (VLAs and benchmarks) which have been explicitly excluded from this repository to avoid enormous repository sizes and duplication.

### 1. Clone this Repository
First, clone this repository to your cluster:
```bash
git clone <your-github-repo-url>
cd DisentangledVLA
```

### 2. Clone Third-Party Dependencies
You must manually clone the required submodules into their respective directories:

**LIBERO Benchmark**
```bash
mkdir -p benchmarks
cd benchmarks
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd ..
```

**Octo Model**
```bash
mkdir -p vlas
cd vlas
git clone https://github.com/octo-models/octo.git
cd ..
```

**OpenVLA-OFT (Custom Fork)**
```bash
cd vlas
git clone https://github.com/moojink/openvla-oft.git
cd ..
```

## Environment Setup (Docker)

This project uses Docker to ensure perfect reproducibility across different clusters (like Slurm).
We provide two separate Dockerfiles:
- `Dockerfile` (OpenVLA and Octo training)
- `Dockerfile.smolvla` (SmolVLA cache builder)

To build the primary training environment:
```bash
docker build -t openvla_worker .
```

To run the container interactively with GPU support:
```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace/DisentangledVLA \
    -w /workspace/DisentangledVLA \
    openvla_worker bash
```

## Project Structure
- `scripts/`: Main execution scripts for building caches and training projectors.
- `src/projectors/`: Architecture implementations (e.g., FlowTransformerProjector).
- `utils/`: Dataloading and cache extraction utilities.

## Uploading to GitHub

If you haven't uploaded this project to GitHub yet, follow these exact steps on your local machine:

1. **Initialize Git** (if not already done)
   ```bash
   git init
   ```
2. **Add Files**
   The `.gitignore` has already been configured to completely ignore heavy checkpoints (`.pt`), logs, weights, wandb runs, and the third-party submodules.
   ```bash
   git add .
   ```
3. **Commit**
   ```bash
   git commit -m "Initial commit of DisentangledVLA codebase"
   ```
4. **Create a Repo on GitHub**
   Go to GitHub, create a new empty repository (do NOT initialize it with a README or .gitignore).
5. **Push**
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/DisentangledVLA.git
   git push -u origin main
   ```
