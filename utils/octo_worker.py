"""
OctoWorker — Persistent Octo subprocess for video probes during projector training.

Why a subprocess?
  JAX and PyTorch both want exclusive CUDA contexts. Running them in the same
  process causes conflicts. The subprocess owns JAX/Octo; the main training
  process owns PyTorch/projector. They communicate via multiprocessing Queues.

Usage:
    from utils.octo_worker import OctoWorker

    worker = OctoWorker(octo_model_path="hf://rail-berkeley/octo-base-1.5")
    worker.start()

    # --- Projector probe ---
    # Returns readout_action embedding (768,) consistent with build_octo_cache.py
    # NOTE: pass RAW image (no flip) — cache was built on raw HDF5 images
    emb_fn = worker.make_emb_fn()   # callable(image_pil, instruction) → Tensor (1, D)

    # --- GT Octo probe ---
    # Returns a full action chunk (chunk_size, 7) from Octo's own action head
    # NOTE: pass [::-1,::-1]-rotated image — matches Octo training preprocessing
    gt_fn = worker.make_gt_fn(unnorm_key="libero_spatial_no_noops")

    worker.stop()

Notes on image preprocessing:
  - embed() / make_emb_fn(): NO image rotation — must match build_octo_cache.py
    which feeds raw HDF5 images directly to Octo.
  - act_gt() / make_gt_fn(): [::-1,::-1] 180° rotation — matches how
    get_octo_action() preprocesses live env images in run_libero_eval.py.

Notes on VRAM:
  - Octo-base ≈ 3 GB. If OOM, pass use_cpu=True — ~150 ms per image on CPU,
    acceptable for probe cadence (every 5k steps).
"""

import multiprocessing as mp
import numpy as np
import os


# ---------------------------------------------------------------------------
# Worker process entry point (runs in the child — owns JAX)
# ---------------------------------------------------------------------------

def _octo_worker_fn(octo_model_path: str, req_q: mp.Queue, resp_q: mp.Queue,
                   use_cpu: bool, libero_action_stats: dict | None):
    """
    Child process: loads Octo once then serves two request types in a loop.

    Request dict keys:
      type="embed"  → image_np (H,W,3 uint8), instruction (str)
                       → (768,) float32 readout_action mean-pool embedding
      type="act"    → image_np (H,W,3 uint8), instruction (str),
                       unnorm_key (str), seed (int)
                       → (chunk_size, 7) float32 action chunk
    Shutdown: None in req_q → puts None and exits.
    """
    # Force CPU-only JAX if requested (avoids VRAM contention).
    # Even when use_cpu=False, if there isn't enough free VRAM for Octo (~3 GB)
    # alongside the PyTorch training process, JAX will OOM at cuDevicePrimaryCtxRetain.
    # Setting use_cpu=True in OctoWorker() avoids this entirely; probes run every
    # 5k steps so CPU inference latency (~2-5 s/image) is acceptable.
    if use_cpu:
        # Use os.environ directly (not setdefault) to ensure these override anything
        # inherited from the parent PyTorch process environment.
        os.environ["JAX_PLATFORMS"] = "cpu"          # JAX >=0.4.1 canonical env var
        os.environ["JAX_PLATFORM_NAME"] = "cpu"      # older alias, kept for compat
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # Best-effort GPU sharing: disable pre-allocation and cap to 20% of VRAM.
        # If PyTorch leaves less than that free, JAX will still OOM — use use_cpu=True.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import jax
        import numpy as np
        from PIL import Image
        from octo.model.octo_model import OctoModel

        print(f"[OctoWorker] Loading Octo from {octo_model_path} …", flush=True)
        model = OctoModel.load_pretrained(octo_model_path)
        octo_dim = model.module.octo_transformer.token_embedding_size
        action_horizon = model.module.heads["action"].action_horizon
        print(f"[OctoWorker] Ready. dim={octo_dim} | action_horizon={action_horizon} | devices: {jax.devices()}", flush=True)
        resp_q.put(("ready", octo_dim, action_horizon))

    except Exception as e:
        resp_q.put(("error", e))
        return

    # Build unnorm statistics for sample_actions().
    # If LIBERO min/max stats are provided, derive mean/std from them so we can use
    # normalization_type=NORMAL without needing a LIBERO-finetuned checkpoint:
    #   mean = (max + min) / 2    → correct center of LIBERO action range
    #   std  = (max - min) / 4   → maps ±2σ of N(0,1) output to [min, max]
    # This is the same approximation used by run_libero_eval when the model lacks
    # the LIBERO key in dataset_statistics (the mask passes gripper through unchanged).
    if libero_action_stats is not None:
        _lo  = np.array(libero_action_stats["min"],  dtype=np.float32)
        _hi  = np.array(libero_action_stats["max"],  dtype=np.float32)
        _msk = np.array(libero_action_stats["mask"], dtype=bool)
        _unnorm_stats = {
            "mean": (_lo + _hi) / 2.0,
            "std":  (_hi - _lo) / 4.0,
            "mask": _msk,
        }
        print(f"[OctoWorker] unnorm stats: derived mean/std from LIBERO min/max "
              f"(on-the-fly, no fine-tuned checkpoint needed)", flush=True)
    else:
        # Fallback: use model's own dataset_statistics (requires a fine-tuned checkpoint)
        _ds = model.dataset_statistics
        _resolved_key = None
        for _candidate in list(_ds.keys()):
            if "libero" in _candidate.lower():
                _resolved_key = _candidate
                break
        if _resolved_key is None:
            _resolved_key = next(iter(_ds))
        _entry = _ds[_resolved_key]
        _unnorm_stats = _entry["action"] if "action" in _entry else _entry
        print(f"[OctoWorker] unnorm stats: using model's '{_resolved_key}' dataset_statistics",
              flush=True)

    # Persistent obs history for GT probe (window_size=2)
    _prev_obs_img: np.ndarray | None = None

    while True:
        item = req_q.get()
        if item is None:            # shutdown sentinel
            resp_q.put(None)
            break

        try:
            req_type = item["type"]

            if req_type == "embed":
                # ----------------------------------------------------------
                # Projector probe embedding — NO image rotation.
                # Must match build_octo_cache.py which uses raw HDF5 images.
                # ----------------------------------------------------------
                image_np  = item["image"]      # (H, W, 3) uint8, raw env image
                instr     = item["instruction"]

                img_256 = np.array(Image.fromarray(image_np).resize((256, 256)))
                batch_imgs = img_256[np.newaxis, np.newaxis]  # (1, 1, 256, 256, 3)

                obs = {
                    "image_primary":     batch_imgs,
                    "timestep_pad_mask": np.ones((1, 1), dtype=bool),
                }
                task = model.create_tasks(texts=[instr])
                transformer_out = model.run_transformer(
                    observations=obs,
                    tasks=task,
                    timestep_pad_mask=obs["timestep_pad_mask"],
                    train=False,
                )
                readout = transformer_out["readout_action"].tokens  # (1,1,n_tokens,D)
                emb = np.array(readout.mean(axis=(0, 1, 2)), dtype=np.float32)  # (D,)
                resp_q.put(("ok", emb))

            elif req_type == "act":
                # ----------------------------------------------------------
                # GT Octo probe — full action inference with sample_actions().
                # Applies [::-1,::-1] rotation to match Octo's training dist.
                # Uses 2-frame history to match Octo's window_size=2.
                # unnorm_stats is passed directly as a numpy dict (min/max/mask)
                # loaded from the LIBERO dataset_statistics.pt — bypasses Octo's
                # built-in Open X-Embodiment statistics entirely.
                # ----------------------------------------------------------
                image_np    = item["image"]        # (H, W, 3) uint8, raw env image
                instr       = item["instruction"]
                seed        = item.get("seed", 0)
                reset       = item.get("reset", False)

                if reset:
                    _prev_obs_img = None

                # Apply 180° rotation to match run_libero_eval preprocessing
                img_rot = image_np[::-1, ::-1]
                img_256 = np.array(Image.fromarray(img_rot).resize((256, 256)))  # (256,256,3)

                # Build 2-frame window: [prev, current] (pad with current on first step)
                if _prev_obs_img is None:
                    _prev_obs_img = img_256
                imgs_window = np.stack([_prev_obs_img, img_256], axis=0)  # (2,256,256,3)
                _prev_obs_img = img_256

                window_size   = 2
                observations = {
                    "image_primary":     imgs_window[np.newaxis],  # (1,2,256,256,3)
                    "timestep_pad_mask": np.ones((1, window_size), dtype=bool),
                    "timestep":          np.arange(window_size, dtype=np.int32)[np.newaxis],
                    "task_completed":    np.zeros((1, window_size, action_horizon), dtype=np.float32),
                    "pad_mask_dict": {
                        "image_primary": np.ones((1, window_size), dtype=bool),
                        "timestep":      np.ones((1, window_size), dtype=bool),
                    },
                }
                task = model.create_tasks(texts=[instr])

                action = model.sample_actions(
                    observations=observations,
                    tasks=task,
                    unnormalization_statistics=_unnorm_stats,
                    rng=jax.random.PRNGKey(seed),
                )
                # action: (1, chunk_size, 7) → (chunk_size, 7) float32
                action_np = np.array(action[0], dtype=np.float32)
                resp_q.put(("ok", action_np))

            elif req_type == "reset_history":
                _prev_obs_img = None
                resp_q.put(("ok", None))

            else:
                resp_q.put(("error", ValueError(f"Unknown request type: {req_type}")))

        except Exception as e:
            resp_q.put(("error", e))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class OctoWorker:
    """
    Manages a persistent JAX/Octo subprocess for use during PyTorch training.

    Two modes:
      embed()   — extracts readout_action token embedding, consistent with
                  build_octo_cache.py (raw image, no rotation).
      act_gt()  — runs full Octo action inference via sample_actions() with
                  180° image rotation and 2-frame history, matching
                  run_libero_eval.py evaluation preprocessing.
    """

    def __init__(self, octo_model_path: str = "hf://rail-berkeley/octo-base-1.5",
                 use_cpu: bool = False, libero_stats_path: str | None = None,
                 libero_suite_key: str = "libero_spatial_no_noops"):
        self.octo_model_path   = octo_model_path
        self.use_cpu           = use_cpu
        self.libero_stats_path = libero_stats_path
        self.libero_suite_key  = libero_suite_key
        self._req_q  = None
        self._resp_q = None
        self._proc   = None
        self.octo_dim: int | None = None
        self.action_horizon: int | None = None

    def start(self):
        """Spawn the child process and block until Octo is loaded."""
        import torch
        # Load LIBERO action stats in the main process (torch available here)
        _libero_action_stats = None
        if self.libero_stats_path is not None:
            try:
                _all = torch.load(self.libero_stats_path, map_location="cpu")
                _entry = _all.get(self.libero_suite_key, _all.get(self.libero_suite_key.replace("_no_noops", ""), None))
                if _entry is not None and "action" in _entry:
                    import numpy as _np
                    _libero_action_stats = {k: _np.array(v) for k, v in _entry["action"].items()}
                    print(f"[OctoWorker] Loaded LIBERO action stats from '{self.libero_stats_path}' "
                          f"(key: {self.libero_suite_key})", flush=True)
                else:
                    print(f"[OctoWorker] ⚠️  Key '{self.libero_suite_key}' not found in {self.libero_stats_path}."
                          f" Will fall back to model's own dataset_statistics.", flush=True)
            except Exception as e:
                print(f"[OctoWorker] ⚠️  Could not load stats from {self.libero_stats_path}: {e}", flush=True)
        ctx = mp.get_context("spawn")   # clean CUDA context, no inherited JAX state
        self._req_q  = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._proc   = ctx.Process(
            target=_octo_worker_fn,
            args=(self.octo_model_path, self._req_q, self._resp_q, self.use_cpu,
                  _libero_action_stats),
            daemon=True,
        )
        self._proc.start()
        status, *payload = self._resp_q.get(timeout=300)
        if status == "error":
            self._proc.terminate()
            raise RuntimeError(f"[OctoWorker] Load failed: {payload[0]}") from payload[0]
        self.octo_dim, self.action_horizon = payload
        print(f"[OctoWorker] Ready. octo_dim={self.octo_dim} action_horizon={self.action_horizon}", flush=True)

    def _call(self, req: dict, timeout: int = 120):
        """Send a request dict and return the response payload."""
        if self._proc is None or not self._proc.is_alive():
            raise RuntimeError("[OctoWorker] Worker not running. Call start() first.")
        self._req_q.put(req)
        status, payload = self._resp_q.get(timeout=timeout)
        if status == "error":
            raise RuntimeError(f"[OctoWorker] Request failed: {payload}") from payload
        return payload

    # ── Embedding (projector probe) ─────────────────────────────────────────

    def embed(self, image_np: np.ndarray, instruction: str) -> np.ndarray:
        """
        Extract readout_action embedding for a single raw (un-rotated) image.
        Returns (octo_dim,) float32.  Consistent with build_octo_cache.py.
        """
        return self._call({"type": "embed", "image": image_np, "instruction": instruction})

    def make_emb_fn(self):
        """
        Returns callable: emb_fn(image_pil, instruction) → torch.Tensor (1, D) on CPU.
        The PIL image is converted to raw numpy WITHOUT applying any rotation,
        matching the preprocessing used when building the Octo cache.
        """
        import torch

        worker = self

        def emb_fn(image_pil, instruction: str):
            import numpy as np
            image_np = np.array(image_pil.convert("RGB"), dtype=np.uint8)
            emb_np   = worker.embed(image_np, instruction)
            return torch.from_numpy(emb_np).unsqueeze(0)   # (1, D)

        return emb_fn

    # ── GT action inference ─────────────────────────────────────────────────

    def reset_gt_history(self):
        """Reset the 2-frame history buffer in the child process (call at episode start)."""
        self._call({"type": "reset_history"})

    def act_gt(self, image_np: np.ndarray, instruction: str,
               seed: int = 0, reset: bool = False) -> np.ndarray:
        """
        Run full Octo action inference on a single raw env image.
        Unnormalization uses model.dataset_statistics['action'] resolved at worker
        startup — identical to get_octo_action() in openvla_utils.py.
        """
        return self._call({
            "type": "act", "image": image_np, "instruction": instruction,
            "seed": seed, "reset": reset,
        })

    def make_gt_fn(self, seed: int = 0):
        """
        Returns callable: gt_fn(image_np, instruction, reset=False) → (action_horizon, 7) np.ndarray.
        """
        worker = self

        def gt_fn(image_np: np.ndarray, instruction: str, reset: bool = False):
            return worker.act_gt(image_np, instruction, seed=seed, reset=reset)

        return gt_fn

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def stop(self):
        """Send shutdown sentinel and wait for child to exit."""
        if self._proc is not None and self._proc.is_alive():
            self._req_q.put(None)
            self._resp_q.get(timeout=30)
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None
        print("[OctoWorker] Stopped.", flush=True)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
