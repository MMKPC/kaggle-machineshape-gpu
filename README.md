# kaggle-machineshape-gpu

Force a specific **GPU type** when you push a Kaggle kernel — something the Kaggle CLI can't do.

*By [Matthew Mitchell — MMKPC Studios](https://github.com/MMKPC).*

```bash
python push_with_shape.py ./my-kernel-folder NvidiaTeslaT4
```

## The problem

The Kaggle CLI (through at least 1.7.x) has **no way to choose your GPU**. `kaggle kernels push`
lets Kaggle assign one for you — and it frequently hands you a **Tesla P100**. That's a problem because:

- **Some competitions ban the P100 at submit time.** Your notebook previews fine, then the *scored*
  submission is rejected with `Your Notebook cannot use P100 GPUs in this competition.`
- **Current PyTorch builds dropped P100 (Pascal / `sm_60`) support.** Even where it's allowed, a CUDA
  op fails at runtime with `no kernel image is available for execution on the device`.

Either way, a GPU run that looked healthy in preview fails at scoring — and there's no CLI flag to fix it.

## The fix

Kaggle's REST endpoint `POST /api/v1/kernels/push` accepts a `machineShape` field that the CLI never sends.
Push through it directly and your kernel — including the automatic re-run at scoring time for code
competitions — lands on the GPU you asked for.

```bash
python push_with_shape.py <kernel_folder> [machineShape]
```

- `<kernel_folder>` — a folder containing `kernel-metadata.json` and the notebook it points to
  (the same layout `kaggle kernels push` expects).
- `[machineShape]` — one of `NvidiaTeslaT4` (default), `NvidiaTeslaP100`, `NvidiaL4`, `NvidiaTeslaV100`.

`NvidiaTeslaT4` (`sm_75`) is the safe default: allowed in competitions that ban the P100, and supported
by stock PyTorch, so you usually don't need to bundle a custom torch build.

## Auth

Either works — nothing is hard-coded, credentials are read at runtime:

- `~/.kaggle/kaggle.json` (the standard Kaggle credentials file), **or**
- environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

## Requirements

- Python 3.8+
- `requests` (`pip install requests`)

## Notes

- This is infrastructure, not magic: it changes *which GPU* your kernel runs on, nothing about the code.
- `machineShape` names come from Kaggle's own API; the list above are the ones seen in practice. If Kaggle
  adds or renames tiers, pass the new string straight through.
- Found and hardened while competing on a code competition where the P100 ban silently failed scored runs.

## Author

Built by **Matthew Mitchell** — MMKPC Studios · [github.com/MMKPC](https://github.com/MMKPC).
Found and hardened while competing on a Kaggle code competition where the P100 ban silently failed scored runs.

## License

MIT — see [LICENSE](LICENSE). The MIT terms require this copyright/attribution to travel with any copy or substantial portion.
