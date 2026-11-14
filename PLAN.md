# PixelTrainer — Project Plan (CIA 3)

Status: planning. Decisions agreed on 2026-11-13 are marked **[agreed]**. Items marked **[verify in Phase N]** are facts not yet measured; they must be confirmed in that phase before code depends on them. Facts marked **Measured** were confirmed in Phase 0.

---

## 1. Scope

### 1.1 Agreed decisions
- **Game:** Pokémon Red (USA/Europe), DMG `.gb`, run in PyBoy. Not Yellow (Pikachu follower breaks tile/NPC detection). **[agreed]**
- **Main goal:** Pallet Town → Viridian City via Route 1 (wild battles only). **[agreed]**
- **Stretch goal:** Viridian City north exit → Route 2 → through Viridian Forest (trainer battles, line-of-sight). Started from a separate save state where Oak's Parcel is already delivered, so the old man no longer blocks the path. **[agreed]**
- **Start state:** intro played once by hand, starter chosen, rival battle done, saved as `.state`. Every run loads a state. **[agreed]**
- **RAM use:** RAM is read (a) offline, to auto-label training data, and (b) by the evaluator, to score perception and task success. The agent's decision loop never reads RAM. The report states this explicitly. Training frames and test frames come from separate runs. **[agreed]**
- **Screen-state model:** tiny CNN (kept because the synopsis promises it), trained on RAM-derived labels. **[agreed]**

### 1.2 Out of scope
Items, healing at Pokémon Centers, switching Pokémon, catching, saving the game, Pokémon Center/Mart interactions beyond dismissing dialogue, any goal past Viridian Forest.

### 1.3 Definition of done
1. `python -m pixeltrainer.run --state route1_start.state --goal viridian` plays to Viridian City with a live side-by-side view.
2. `python -m evaluation.run_eval` runs N seeded episodes and writes the accuracy and task report (tables + plots).
3. A test asserts that no module under `pixeltrainer/` imports the RAM reader.
4. Report (PDF) and a recorded demo video, as backup for the live demo.

---

## 2. Environment and assets

| Item | Source | Notes |
|---|---|---|
| Python 3.11.9 | installed | |
| numpy 2.2.4, torch 2.12.1 | installed | CPU is enough; all models are tiny. |
| `pyboy` | `pip install pyboy` | v2.x API: `PyBoy(rom, window="null")`, `tick()`, `button()`, `screen.ndarray`, `memory[]`, `save_state/load_state`. |
| `opencv-python` | `pip install opencv-python` | phase correlation, template matching, live view window, video writing. |
| `pokered.gb` | user's own cartridge dump | At `roms/pokered.gb`. Title `POKEMON RED`, boots in PyBoy. SHA1 `1ed352e5…` is not the clean dump (`ea9bcae6…`): file is tagged `[S][BF]` (patched). RAM addresses should be unaffected, but Phase 0 must confirm them on this file. Never committed. |
| `pokered.sym` | pret/pokered `symbols` branch, at `roms/pokered.sym` | Gives exact WRAM addresses. Loaded with `PyBoy(rom, symbols=...)`; `symbol_lookup` works. |
| Reference sprites/font | captured from the running game (preferred) or pret `gfx/` PNGs | One template per class. |

`.gitignore` covers: `*.gb`, `*.state`, `*.sav`, `data/`, `runs/`, `assets/captured/`.

---

## 3. Architecture

```
            ┌─────────────────────────── agent loop (pixels only) ───────────────────────────┐
frame ──►   ScreenState CNN ──► overworld ──► TileClassifier + NPCDetector ──► Registration ──► WorldMap ──► Planner (A*) ──► button
(160×144)                 ├──► battle    ──► BattlePerception (sprite ID, HP bars, OCR) ──► BattlePolicy ──► button
                          ├──► dialogue  ──► press A until state changes
                          └──► menu      ──► press B (or navigate when the battle menu is expected)
            └────────────────────────────────────────────────────────────────────────────────────┘
                     │ every perception output is logged per frame
                     ▼
            Evaluator ◄── RAM truth (truth.py, eval only)
```

### 3.1 Repository layout
```
pixeltrainer/            # the agent. Must NOT import evaluation/ (enforced by tests/test_ram_boundary.py)
  emu.py                 # PyBoy wrapper: reset(state), press(button, frames), frame(), wait_until_stable()
  perception/
    screen_state.py      # CNN: overworld / battle / dialogue / menu (+ "transition" for fades)
    grid.py              # frame → 10×9 grid of 16×16 cells, player cell location
    tiles.py             # cell classifier: walkable, wall, grass, water, door, ledge_{s,e,w}, sign, npc
    npc.py               # NPC detection + facing direction
    battle.py            # enemy sprite ID, HP bar reader, battle menu/cursor state
    ocr.py               # fixed-font 8×8 OCR
  mapping/
    registration.py      # frame-to-frame shift (phase correlation) + transition detection
    worldmap.py          # global grid, segments (one per building/area), position estimate
  planning/
    astar.py             # A* with terrain costs and one-way ledges
    explore.py           # frontier exploration biased toward goal direction
  policy/
    battle_policy.py     # random / greedy / type-aware expected-damage
    gamedata.py          # Gen 1 type chart, move table, species→types (static tables)
  agent.py               # main loop, state machine
  viz.py                 # side-by-side live view + mp4 recording
  run.py                 # CLI entry
evaluation/
  truth.py               # RAM reader (addresses from pokered.sym)
  run_eval.py            # seeded episodes, logs perception vs truth
  report.py              # tables, plots
tools/
  check_rom.py           # SHA1 check
  make_states.py         # helper to save start states while playing by hand
  record_dataset.py      # scripted/random play, saves frames + RAM labels
  label_cells.py         # one-time labelling of unique 16×16 cells (pre-filled from RAM collision)
  capture_templates.py   # captures species sprites, font glyphs, NPC sprites from live frames
tests/
assets/                  # templates (committed only if captured by us; ROM/state never)
```

### 3.2 Core facts the design relies on (measured in Phase 0 by `tools/probe_geometry.py`, raw numbers in `runs/probe/probe.json`)
- Screen is 160×144 = 20×18 tiles of 8×8. **Measured:** the 16×16 cell grid is aligned to screen (0,0) when standing still (26 unique cells at offset (0,0) vs ≥ 31 at any other 4 px offset; scroll registers SCX = 0, SCY = 16). View = 10×9 cells.
- **Measured:** player sprite at screen (x 64, y 60), i.e. cell column 4, row 4, drawn 4 px above the cell.
- **Measured:** one step = 16 frames of movement (18–19 when the first press also turns the player), and scrolls exactly 16 px (exact-match shift 16 px on every step, ≥ 97% of pixels agree outside the player cell; the rest is animated flowers/water). `cv2.phaseCorrelate` gives ±16.0 ± 0.1 px with response 0.56–0.81. Sign: walking right moves content left (phase dx ≈ −16).
- **Measured:** a blocked step scrolls 0 px (phase response 1.0 at shift 0); holding the button keeps it blocked.
- **Measured:** the Pallet → Route 1 border is seamless (16 px shift, 100% match, no fade; map 0 at (10,0) → map 12 at (10,35)). Door entry/exit fades through a full black screen (mean 0), taking ~65–75 frames until a stable frame. On these, phase-correlation response drops to ≤ 0.07, so a low response marks a transition.
- **Measured:** the default PyBoy palette has exactly 4 grey levels: 0, 85, 153, 255 (RGB channels equal).
- Battle: enemy front sprite in a fixed box top-right, enemy HP bar top-left (48 px wide), player HP shown as numbers. **[verify in Phase 5]**
- Route 1 ledges face south. Going north, the agent must path around them. **[verify in Phase 3]**
- RAM addresses (from pokered.sym; values checked against the screen in 0.6, see `runs/truth_check.md`): `wCurMap` D35E, `wYCoord` D361, `wXCoord` D362, `wIsInBattle` D057, `wEnemyMonSpecies` CFE5, `wEnemyMonHP` CFE6, `wEnemyMonLevel` CFF3, `wEnemyMonMaxHP` CFF4, `wBattleMonHP` D015, `wBattleMonMaxHP` D023, sprite tables `wSpriteStateData1` C100 / `wSpriteStateData2` C200 (16 × 16 bytes, player = slot 0; the player's map coords in C200 are not maintained, use `wXCoord`/`wYCoord`), `wTilesetCollisionPtr` D530, `wGrassTile` D535, `wTileMap` C3A0. `evaluation/truth.py` reads them by symbol name.
- **Measured** map IDs: Pallet 0, Viridian 1, Route 1 12 (0x0C), Oak's lab 40 (0x28). Route 2 0x0D and Viridian Forest 0x33 **[verify in Phase 6]**.
- **Measured** species are internal indices: Charmander 176 (0xB0), Pidgey 36 (0x24). Moves: Scratch 10, Growl 45.
- **Measured** screen-state signals: `wFontLoaded` = 1 while an overworld text box is open, 0 otherwise (also 0 in battle); `wIsInBattle` = 1 in a wild battle. `wTextBoxID` is not useful (1 in plain overworld). Menu signals still to be found in 2.1.

---

## 4. Module design

### 4.1 Emulator harness (`emu.py`)
- Headless by default, `set_emulation_speed(0)` for evaluation; real-time with window for the demo.
- `walk(direction)`: hold the direction until the screen starts scrolling (>3% of pixels change outside the player sprite), then release; Gen 1 finishes a started step, so this is exactly one step (tested against RAM, including turn-then-step). No scroll within 24 frames = blocked. `press(button)`: hold 4 frames, release. After either, `wait_until_stable()` = tick until no *new* screen (player sprite masked) appears for 8 frames and the frame is not blank, or 600 frames pass. Animations (flowers, water, party icons, text arrow) cycle through screens already seen; printing text makes a new screen every few frames, so half-printed text is never "stable"; fades sit on a blank screen. Median 28 frames per decision, none hit the cap on 2 test runs.
- Seeds: Gen 1 RNG depends on timing, so episode variety comes from idling a random 0–300 frames after loading the state.

### 4.2 Screen-state CNN (`screen_state.py`)
- Input: frame downscaled to 80×72 grey. Network: 3 conv layers + 1 linear layer, under 50k parameters.
- Classes: overworld, battle, dialogue, menu, transition.
- Labels from RAM: `wIsInBattle`, `wFontLoaded` (see §3.2); menu signal found in 2.1. Fallback: hand-label ~500 frames with a keypress labeller.
- Target: ≥99% accuracy on held-out runs. Baseline for the report: a hand-written heuristic (text-box border detection).

### 4.3 Grid and tile classification (`grid.py`, `tiles.py`)
- Cut the stable frame into 10×9 cells of 16×16.
- Labelling: hash every unique cell seen in recorded data (expected a few hundred for Pallet/Route 1/Viridian). Label each unique cell once in `label_cells.py`. Walkable/blocked is pre-filled from RAM collision data, so hand work is mostly grass/water/door/ledge.
- Two classifiers, both reported:
  1. **Lookup baseline:** exact-hash → label. Near-perfect on seen cells, fails on unseen ones.
  2. **CNN:** small conv net on 16×16 cells with light augmentation. Tested on cells from Viridian Forest that it never trained on, to show generalisation.
- Walkable/blocked is scored against RAM collision; the other classes are scored against the hand labels.

### 4.4 NPC detection (`npc.py`)
- NPC sprites are 16×16 with four facing frames. Templates captured once with labels from the RAM sprite table.
- Template matching per cell (sprites are cell-aligned when stationary). Output: cell, facing direction, sprite type.
- Line of sight: cells along the facing ray up to K cells (K default 4, tunable) get a high planning cost when the NPC looks like a trainer.
- Scored against the RAM sprite table: detection precision/recall and facing accuracy.

### 4.5 Registration and map building (`registration.py`, `worldmap.py`)
- Mask out the player cell and detected NPCs, then `cv2.phaseCorrelate` between the previous and current stable frames. Expected result: shift of 0 or ±16 px on one axis.
- The commanded direction gives a prior. A measured shift of 0 after a move command means "blocked", which also marks the target cell as a wall (self-correcting map).
- Transition detection: screen-state "transition" (fade) or a near-total content change. Starts a new map segment. Doors link segments.
- `WorldMap`: dict `(segment, x, y) → label, confidence, last_seen`. Newer observations overwrite older ones only when confident.
- Evaluation: per-step odometry error, and absolute position error in tiles within each segment after aligning the origin. Outdoor maps are one segment for the agent but several maps in RAM, so the evaluator converts RAM coordinates using the map-connection offsets.

### 4.6 Planning (`astar.py`, `explore.py`)
- 4-connected A*. Costs: walkable 1, grass 3 (tunable), NPC sight-line 20, door 1 only when it is the goal, wall/water/NPC impassable, unknown 2 (optimistic).
- Ledges: passable only in the jump direction, as an edge from the cell above to the cell two below.
- Goal specification: compass direction plus a visual landmark (e.g. Pokémon Center roof template). The agent explores the frontier cell that best combines distance and progress in the goal direction. The agent declares success when it sees the landmark. The evaluator independently confirms via `wCurMap`.
- Replan after every step (the grid is tiny, so this is cheap).

### 4.7 Battle perception (`battle.py`, `ocr.py`)
- **OCR:** text is aligned to the 8×8 tile grid in a fixed font, so each cell is matched against the glyph templates (exact match first, then nearest by Hamming distance). Reads battle text, move names, PP, and the player's HP numbers.
- **Menu state:** find the "▶" cursor glyph to know which option is selected, then move it with the D-pad.
- **Enemy species:** read only after the slide-in animation (stable frames). Normalised cross-correlation against species templates, with a small embedding model as a second method compared in the report. Route 1 and the forest need ~10 species, but templates for all 151 make the test honest.
- **HP bars:** threshold the bar region, count filled pixels out of 48, giving the HP fraction. Scored against RAM HP / max HP.

### 4.8 Battle policy (`battle_policy.py`, `gamedata.py`)
- Static Gen 1 tables: type chart (including Gen 1 quirks), moves (power, type, accuracy), species → types.
- Expected damage = power × STAB × effectiveness × accuracy. Status moves get a small fixed value, used only when no damaging move exists.
- Baselines: random move; greedy = highest base power, ignoring types.
- Decision text for the live view, e.g. `Ember vs Caterpie: 2× → exp 60`.

### 4.9 Agent loop (`agent.py`)
State machine on the screen-state output:
- overworld → perceive, update map, plan, step
- battle → read the menu; at the main menu choose FIGHT, then the chosen move; otherwise press A to advance text
- dialogue → press A
- menu (unexpected) → press B
- transition → wait

Watchdogs: if the same state repeats for N decisions without progress, take a random escape action and log it.

### 4.10 Live view (`viz.py`)
OpenCV window, about 960×432. Left: the game at 3×. Right: tile labels overlaid on the screen, the growing map with the agent and planned path, and the last battle reasoning line. Optional mp4 recording.

---

## 5. Evaluation plan

| Metric | Truth source | Target |
|---|---|---|
| Screen-state accuracy (+ confusion matrix) | RAM / hand labels | ≥ 99% |
| Tile accuracy per class, lookup vs CNN, seen vs unseen area | RAM collision + hand labels | walkable ≥ 98% |
| NPC detection P/R, facing accuracy | RAM sprite table | ≥ 95% |
| Position error (tiles), odometry drift per 100 steps | `wXCoord`, `wYCoord`, `wCurMap` | 0 drift on clean runs |
| Species top-1 accuracy | enemy species byte | ≥ 95% |
| HP fraction MAE | RAM HP / max HP | ≤ 2% |
| OCR character accuracy | text tiles in `wTileMap` (C3A0) | ≥ 99.5% |
| Task success rate, steps / shortest path | `wCurMap`, path length on a true map | report |
| Battle win rate: type-aware vs greedy vs random | battle outcome from RAM | report |
| Decisions per second | timer | report |

Protocol: 20 seeded episodes per goal and policy. Perception metrics come from runs not used for training. Ablation: map without the "blocked move" self-correction; planner without grass costs.

---

## 6. Phases and milestones

| # | Phase | Deliverable / acceptance |
|---|---|---|
| 0 | **Setup and verification** | Packages installed, ROM SHA1 checked, start states saved, `truth.py` reads position/map/battle data that matches the screen while playing by hand, all [verify] facts confirmed or corrected in this file. |
| 1 | **Harness and dataset** | `emu.py`, `record_dataset.py`: ~20k labelled frames from scripted random walks + battles across separate train/test runs. |
| 2 | **Screen state + grid + tiles** | CNN ≥ 99%, unique cells labelled, lookup + tile CNN working, accuracy numbers produced. |
| 3 | **Registration + map** | Hand-driven walk Pallet → Viridian builds a correct map with 0 position error against RAM. |
| 4 | **Planner + overworld agent** | Agent reaches Viridian with battles handled by "press A / first move" placeholder. |
| 5 | **Battle perception + policy** | OCR, species, HP, menus; three policies; win-rate numbers. |
| 6 | **NPCs + stretch goal** | NPC/facing detection, sight-line costs, Viridian Forest attempt. |
| 7 | **Live view + evaluation + report** | `viz.py`, `run_eval.py`, `report.py`, demo video, written report. |

| 8 | **Capture front-end + robustness** (§6A) | Simulated degradation, screen localisation, rectification, restoration, quantisation; robustness curve and restoration table. Simulated only, no camera. |

Critical path: 0 → 1 → 2 → 3 → 4 gives a working navigation demo. Phase 5 makes battles intelligent. Phase 8 brings in the taught syllabus (§6A) and the main robustness result. In Phase 6, NPC detection is core (it stops NPC bumps being mapped as walls); sight-lines and the Viridian Forest goal are optional. Order: 0–4, 5, 6.1–6.2, 8, 7, then the stretch goals.

---

## 6A. Syllabus integration

Course MCA509D-5 (`syllabus.pdf`). Taught so far: U1 (T1, T2), U2 (T1, T2), U3-T1. The project uses the main taught topics, and only where they help. Untaught topics (U3-T2 onward: watershed, structure from motion, optical flow, YOLO-style detectors) are not used, except for CNNs, which the synopsis already promises.

Clean emulator frames need little image processing, so one change gives these methods a real job: **simulated degradation**. In `sim:<level>` mode the agent sees a corrupted, camera-like view of the screen and must recover a clean frame before the normal pipeline runs. The RAM truth still scores everything, so no extra labelling is needed. No real camera.

### 6A.1 Capture front-end
```
clean frame ─► evaluation/degrade.py ─► degraded image ─► localise ─► rectify ─► restore ─► quantise ─► 144×160 shade frame ─► rest of the pipeline
               (seeded, levels 0–4)                       (edges,     (inverse    (filters,   (histogram
                                                           lines,      geometric   frequency   thresholds)
                                                           corners)    transform)  domain)
```
- **`evaluation/degrade.py`** (evaluation side: it gets the clean frame; the agent gets only its output). Degradation model `g = h * f + η`:
  - geometric: perspective tilt, rotation, scale; screen placed on a cluttered background;
  - blur: Gaussian defocus, linear motion blur;
  - noise models: Gaussian, salt-and-pepper, Poisson;
  - photometric: gamma, illumination gradient, colour cast (output is RGB);
  - periodic: moiré / scanline pattern.
  Each type alone at strengths 1–4, plus 5 combined levels 0–4.
- **`pixeltrainer/capture/localise.py`.** Sobel/Canny edges, contour polygon fit for the screen quad; Hough-line fallback; Harris corner refinement.
- **`pixeltrainer/capture/rectify.py`.** Perspective transform from the 4 corners back to 160×144 (bilinear interpolation).
- **`pixeltrainer/capture/restore.py`.** Compared by PSNR/SSIM and downstream accuracy:
  - spatial smoothing: mean, Gaussian, median, adaptive median (salt-and-pepper);
  - sharpening: Laplacian / unsharp masking after deblurring;
  - frequency domain: Gaussian/Butterworth low-pass vs the spatial versions; notch filter on moiré peaks in the FFT magnitude spectrum; Wiener deconvolution for motion blur (PSF known in sim);
  - point operations: gamma correction, histogram equalisation / CLAHE for uneven lighting;
  - temporal: average N frames while the game is idle (the screen is static between decisions, so noise falls by about √N);
  - optional: Haar wavelet soft-threshold denoising (U2-T2 intro to wavelets).
- **`pixeltrainer/capture/quantise.py`.** RGB → grey, then multi-level thresholding at histogram valleys (multi-Otsu) back to 4 shades.

### 6A.2 Topic map (taught topics only)

| Topic | Unit | Where it is used | Why it helps | Compared / metric |
|---|---|---|---|---|
| Correlation and convolution | U1-T1 | NCC template matching: sprites, font glyphs, landmarks; every filter in `restore.py` | Core matcher for all fixed-art recognition | species top-1, OCR accuracy |
| Sampling and quantisation, resolution | U1-T2 | RGBA → 4 shades in `emu.frame()`; 80×72 CNN input; `quantise.py` | One compact input format for every module | shade accuracy after restoration |
| Neighbours, connectivity, distance | U1-T2 | 4-connected A* with Manhattan heuristic; connected components for NPC candidates; Hamming distance for glyphs | Correct grid moves; sprite candidates without a learned detector | — |
| Colour models | U1-T2 | Degraded RGB → grey; colour-cast removal | Degraded input is tinted colour | quantisation accuracy |
| Geometric transformations | U1-T2 | Perspective warp in `degrade.py`, inverse warp in `rectify.py`; world-map mosaic by translation | Undoes the tilted view; mosaic shows the explored world | corner error, mosaic seams |
| Gray-level transforms, histogram processing / equalisation | U2-T1 | Gamma correction, CLAHE before quantisation; global intensity histogram for fade detection | Lighting gradients otherwise break fixed thresholds | with/without, per level |
| Degradation model, noise models, restoration in noise | U2-T1 | `degrade.py`; mean/median/adaptive median in `restore.py` | Turns clean input into a real restoration problem | PSNR/SSIM, task success vs level |
| Smoothing and sharpening spatial filters | U2-T1 | Gaussian/median smoothing; Laplacian/unsharp sharpening | Matching needs crisp 8×8 glyph edges | OCR accuracy with/without |
| Fourier transform, frequency-domain filters | U2-T2 | Phase correlation for frame registration; low-pass vs spatial; notch for moiré; Wiener for blur | One-FFT shift estimate; removes periodic noise spatial filters cannot | notch vs median on moiré; spatial vs frequency smoothing |
| Wavelets (intro) | U2-T2 | Optional Haar soft-threshold denoising | Edge-preserving denoising, one more row in the table | PSNR/SSIM |
| Local and global features; detection, description, matching | U3-T1 | ORB keypoints + descriptor matching + RANSAC as a second registration method; global histogram features in the screen-state heuristic | Registration still works under heavy blur where phase correlation drifts | phase correlation vs ORB, per level |
| Point, line and edge detection | U3-T1 | Sobel/Canny screen border, Hough lines, Harris corners in `localise.py` | Finds the screen in a cluttered degraded image | corner error (px), localisation success |
| Thresholding | U3-T1 | HP bar fill (Otsu), text binarisation, multi-level quantisation | HP reading and OCR on noisy input | HP MAE, OCR accuracy |
| Region growing | U3-T1 | Grow the walkable region from the player cell on the world map (reachable set); prune unreachable frontier cells in exploration | Explorer stops targeting cells it can never reach | exploration steps with/without |
| CNNs (synopsis) | — | Screen state, tile classifier | Generalises to unseen tiles and degraded input | vs heuristic / lookup |

Not used: region splitting and merging (no task needs it), and every untaught topic.

### 6A.3 Headline results
1. **Robustness curve:** task success and per-module accuracy against degradation level 0–4, restoration off vs on.
2. **Restoration table:** PSNR/SSIM and downstream accuracy per filter, per degradation type.
3. **Method comparisons:** registration (phase correlation vs ORB + RANSAC), smoothing (spatial vs frequency), screen state (heuristic vs CNN), tiles (lookup vs CNN), species (NCC vs CNN).
4. **World-map mosaic** of the explored area.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| A [verify] fact is wrong (addresses, grid alignment, sprite offset) | Phase 0 checks every one before any dependent code. |
| Dialogue/menu edge cases trap the agent | Watchdog + logging of stuck states. Route 1 has few NPCs. |
| Ledges and one-way paths confuse exploration | Ledge as a directed edge; "blocked" feedback corrects the map. |
| Wild battles too frequent, slowing evaluation | Headless max-speed emulation; grass cost in the planner. |
| Starter faints (blackout) | Counts as a failure in the metrics. Rare at Route 1 levels. Report it. |
| Accusation of RAM cheating | Import-boundary test, separate evaluation module, explicit statement in the report. |
| Live demo fails on presentation day | Recorded demo video as backup. |

---

## 8. Open questions
1. Deadline and rubric for CIA 3. These decide whether Phase 6 is in scope.
2. Report format and page limit.
3. Solo or team project. Affects how phases are split.
