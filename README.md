# SO-101 cube-in-bin — ManiSkill scene

Real-world-matching ManiSkill scene for the SO-101 pick-a-cube-and-place-it-in-the-bin task.
Source of truth for object positions/sizes/colors is `scene_config.yaml`.

## Files

| File | Purpose |
|---|---|
| `scene_config.yaml` | All real-world measurements (positions, sizes, colors, camera pose) |
| `render_scene.py` | Loads YAML, builds the SAPIEN scene, saves `render.png` |
| `requirements.txt` | Python deps |

## Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### macOS (Apple Silicon)

SAPIEN needs MoltenVK for Vulkan. Install via Homebrew (one-time):

```bash
brew install molten-vk vulkan-loader
```

`render_scene.py` auto-detects macOS and points `VK_ICD_FILENAMES` at the Homebrew MoltenVK ICD.

### Linux + NVIDIA CUDA

Nothing extra — the system Vulkan loader finds the NVIDIA driver automatically.

## Render

```bash
python render_scene.py
# writes ./render.png
```

## Iteration loop

1. Edit `scene_config.yaml` (a number, a color, a camera pose).
2. `python render_scene.py`
3. Open `render.png`, compare to your real photo.
4. Repeat.

## Next steps (not in this repo yet)

- Wrap the scene as a proper ManiSkill `BaseEnv` for training (registered env id, observation/action spaces, reward, success conditions).
- Add **domain randomization** over bin color/texture, lighting, camera pose, object positions — rather than matching one real bin texture exactly. See README discussion.
- Tip-over termination + large negative reward in the env (real bin is light and tippable).
