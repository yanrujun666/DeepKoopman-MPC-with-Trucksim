# DDK TruckSim Python MPC

Koopman MPC **V2** (encoder + Koopman dynamics + constrained QP) for **TruckSim**, closed through **Simulink** and a MATLAB Function that calls Python. This repo includes **example checkpoints** under `ckpt/` and **example datasets** under `data/` so you can align paths and run without fetching assets elsewhere.

---

## At a glance

| Layer | Role |
|--------|------|
| **Simulink** (~1 ms) | Vehicle simulation |
| **MATLAB Function** | `MPC/matlab_function_wrapper.m` calls `py.ddk_mpc_sfunction.*` |
| **Python** | `MPC/ddk_mpc_sfunction.py` drives `MPC/koopman_mpc_v2.py` |
| **In** | **6x1** state (relative x, y, yaw, vx, vy, yaw rate; SI) |
| **Out** | **12x1** controls (six torques, six steers; see wrapper for TruckSim-facing units) |

Call-chain details: `MPC/markdown/代码结构说明.md`.

---

## Bundled weights (`ckpt/`)

Example PyTorch checkpoints shipped with the repo:

- `DeepEDMD-Transv2wonorm-hd16-multiset-100e-remote-local-lr1e-4-rollover-0.05pilossv24-0222.pth` — matches the default `param_path` in `MPC/matlab_function_wrapper.m`
- `DeepEDMD-Transv2-hd16-multiset-100e-remote.pth`
- `LinearModel-multiset.pth`

Point `param_path` in the wrapper at the file you want to use.

---

## Bundled data (`data/`)

| Location | Contents |
|----------|----------|
| `data/all/`, `data/all_rollover/`, `data/c1_*` ... | Training-style `.mat` (e.g. `Vehicle_state_trucksim_39d`) |
| `data/ref_traj/all/` | Reference trajectories for MPC (`*_ref.mat`, 0.01 s sampling) |
| `data/ref_traj/gen_ref/` | MATLAB generators (`.m`) plus small demo `.mat` / `.fig` |
| `data/exp_traj_log/` | Example trajectory logs (`vehicle_trajectory_log*.csv`) and comparison plots |

Default `data_path` in `matlab_function_wrapper.m` is under `data/ref_traj/all/` (snake acceleration scenario); change it to any `*_ref.mat` you need.

---

## Repo layout

```
MPC/ Simulink bridge + V2 MPC (Python)
mpc_dk/              Optional training stack; sample conda env
scripts/             convert_dataset_to_ref, plot_trajectory_compare, analyze_mat_formats
ckpt/                Example .pth weights
data/                Example .mat / logs / ref generators
```

---

## Environment

Python **3.9+** with `numpy`, `scipy`, `torch`, `cvxpy`; `matplotlib` for scripts. Start from `mpc_dk/environment.yml` and adjust CUDA/CPU. `ddk_mpc_sfunction.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` when useful for MATLAB/OpenMP coexistence.

---

## Simulink quick setup

1. Prepend the repo `MPC` folder on the MATLAB Python path, e.g. in **InitFcn**:

   ```matlab
   py.sys.path().insert(int32(0), '<repo-root>/MPC');
   ```

2. Use `matlab_function_wrapper.m` in a **MATLAB Function** block (**6x1** in, **12x1** out).
3. Set `param_path` and `data_path` to files under `ckpt/` and `data/ref_traj/all/` (defaults already point at bundled examples if your root matches the wrapper).
4. Tune `Q`, `R`, `I`, horizons, and `decimation` as needed (see comments in the wrapper).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_dataset_to_ref.py` | Training `.mat` to MPC reference format; `--batch`: `data/all` to `data/ref_traj/all/*_ref.mat` |
| `scripts/plot_trajectory_compare.py` | XY: logged CSV vs reference `.mat` |
| `scripts/analyze_mat_formats.py` | Inspect `.mat` structure |

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
