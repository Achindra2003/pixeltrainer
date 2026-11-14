# PixelTrainer

A vision-only agent that plays Pokémon Red from screen pixels. See [Project_Synopsis.md](Project_Synopsis.md) for the idea, [PLAN.md](PLAN.md) for the design and [BUILD.md](BUILD.md) for the task list.

## Setup (Windows, PowerShell)

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python tools\check_rom.py      # expects roms/pokered.gb
.\.venv\Scripts\python -m pytest -q
```

The ROM is not included. Put your own dump of Pokémon Red at `roms/pokered.gb`.

## Rule

Code under `pixeltrainer/` decides from screen pixels only. RAM is read only by `evaluation/`, for labelling and scoring. `tests/test_ram_boundary.py` enforces this.
