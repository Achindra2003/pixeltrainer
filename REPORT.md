# PixelTrainer: a vision-only agent for Pokémon Red

CIA 3, Computer Vision (MCA509D-5), Trimester 5.

## 1. Summary

PixelTrainer plays Pokémon Red in the PyBoy emulator using only the pixels on the screen. Starting in Pallet Town, it walks through Route 1 to Viridian City, fights the wild battles on the way, and stops when it sees the Viridian Poké Center sign. Every decision comes from image analysis: screen-state classification, tile and NPC detection, frame registration into a world map, template matching for sprites and text, and an A* planner over the map the agent builds.

A second mode makes the vision problem harder. In `sim:<level>` mode each frame is first degraded into a camera-like photograph of the screen (tilt, blur, noise, colour cast, uneven light, moiré). The agent must then find the screen, restore it, rectify it and quantise it back to the game's 4 shades before the normal pipeline runs. This is where most of the taught image-processing methods do real work.

Headline results:
- On clean frames the agent reaches Viridian City in **60/60** episodes (20 seeds × 3 battle policies), with 0 map drift.
- Perception components meet their targets on unseen test runs: screen state 99.49%, NPC detection precision 100% and recall 97.5%, species 100%, HP bar MAE 0.63%, OCR 99.99% on drawn text, map position error 0 tiles on 12,212 frames.
- Under simulated capture the agent still succeeds at degradation level 1 in 3/3 seeds with restoration (2/3 without it). It fails from level 2 upward. The first part to break is the screen-state CNN, which was trained only on clean frames.

## 2. Use of RAM (statement)

The agent never reads the emulator's memory. RAM is used in two places only, both outside the agent:
1. **Offline labelling** of training data (screen state, tile walkability, NPC positions and facings, species, HP), from recorded runs 0–14.
2. **Scoring**, by the evaluator (`evaluation/truth.py`), which compares what the agent perceived with the game's true state.

Training frames (runs 0–14) and test frames (runs 15–19) come from separate recordings. `tests/test_ram_boundary.py` fails if any module under `pixeltrainer/` imports the evaluation package or calls a PyBoy memory, tilemap or sprite API. The agent's only input is the 160×144 frame.

## 3. System

```
frame ─► [sim mode: localise ─► restore ─► rectify ─► quantise] ─► screen state (CNN)
            overworld ─► tiles (lookup) + NPCs ─► registration ─► world map ─► explore + A* ─► button
            battle    ─► menu, cursor, species, HP, OCR ─► move policy ─► button
            dialogue / menu / transition ─► A / B / wait
```

| Module | Method | Course topic |
|---|---|---|
| `emu.py` | RGBA screen to 4 shade indices | Sampling and quantisation (U1-T2) |
| `perception/screen_state.py` | Small strided CNN on the full frame; hand-written baseline from 8×8 glyph matches | CNN; correlation (U1-T1) |
| `perception/grid.py`, `tiles.py` | Frame split into a 9×10 grid of 16×16 cells; exact cell-hash lookup; small CNN per cell (3 conv layers, ±1 px shift and noise augmentation) | Neighbourhoods (U1-T2); CNN |
| `perception/npc.py` | Connected components of the darkest shade as candidates, masked NCC against RAM-labelled templates, IoU de-duplication | Connectivity (U1-T2); correlation (U1-T1) |
| `mapping/registration.py` | Phase correlation (FFT), with the pressed direction as a prior | Fourier transform (U2-T2) |
| `mapping/worldmap.py` | Segments, mosaic by translation, wall marking from blocked moves, relocalisation by rare cell keys | Geometric transforms (U1-T2) |
| `planning/explore.py`, `astar.py` | Region growing from the player cell, frontier scoring, 4-connected A* with a Manhattan heuristic | Region growing (U3-T1); distance (U1-T2) |
| `perception/battle.py`, `ocr.py`, `landmark.py` | NCC against 151 species sprites; HP bar fill; 8×8 glyph matching with Hamming fallback; landmark NCC | Correlation, thresholding (U3-T1) |
| `policy/battle_policy.py` | Expected damage = power × STAB × type effectiveness × accuracy | — |
| `capture/localise.py` | Background model subtraction, Hough lines per side, Sobel and edge-model refinement, corners by line intersection | Point, line and edge detection (U3-T1) |
| `capture/restore.py` | Mean, Gaussian, median, adaptive median, unsharp, low-pass, notch, Wiener, gamma, histogram equalisation/CLAHE, frame averaging | Degradation and restoration (U2-T1); frequency filters (U2-T2) |
| `capture/rectify.py`, `quantise.py` | Perspective warp to 160×144; thresholds at the valleys of the smoothed log-histogram | Geometric transforms; histogram processing; thresholding |

Untaught methods (watershed, optical flow, structure from motion, learned detectors) are not used. CNNs are used because the synopsis promised them.

## 4. Method notes

**Screen state.** Five classes: overworld, battle, dialogue, menu, transition. Labels come from RAM plus glyph visibility (spot-checked by hand: 0 disagreements in 200 frames). The CNN has 4 strided convolution layers and 3×3 adaptive pooling, so it keeps where things are (HUD corners versus a bottom text box).

**Registration and map.** One step moves the screen by exactly 16 px. Given the pressed direction, the agent compares the frames shifted by 0, 1 and 2 cells (mean absolute difference over the overlap) and takes the best. If none fits, which happens on noisy degraded frames, phase correlation decides among the same moves. A blocked move (0 px) marks the target cell as a wall. A bump into a detected NPC marks it "npc" at low confidence instead, so the planner avoids it until the tile is seen again. After a gap (battle, menu, door), the frame is registered against the last overworld frame or relocalised against all known segments.

**NPC detection.** Templates come from RAM-labelled crops on runs 0–14. One per-pixel mode is taken per location, then the mode across locations, and pixels where fewer than 90% of locations agree are masked out. This removes transparent pixels and the lower half that tall grass draws over. Candidate boxes are any standing-sprite box that holds part of a darkest-shade component. Limiting components to sprite size lost 69% of NPCs, because outlines join fences and trees.

**Capture front-end.** `evaluation/degrade.py` applies `g = h * f + η` with seeded geometric, blur, noise, photometric and moiré degradations, each at strengths 1–4, combined into levels 0–4 (level 1: slight tilt, defocus, Gaussian noise and colour cast; level 2 adds stronger blur, gamma and an illumination gradient; levels 3–4 add motion blur, salt-and-pepper or Poisson noise and moiré). The agent's chain is: localise the screen, flatten the lighting, Wiener deconvolution with a blind σ (the σ whose rectified result has the tightest shade clusters), rectify, quantise.

## 5. Results

### 5.1 Task and battles (clean frames)

20 seeds per policy (`python evaluation/report.py runs/eval/final`):

| Policy | Reached Viridian | Decisions (mean) | Decisions/s | Battles | Won | Drift per 100 steps | Battle state vs RAM |
|---|---|---|---|---|---|---|---|
| random | 20/20 | 290 | 31.9 | 37 | 81.1% | 0 (1,925 steps) | 99.77% |
| greedy | 20/20 | 206 | 30.9 | 36 | 100% | 0 (1,526 steps) | 99.68% |
| type-aware | 20/20 | 206 | 32.2 | 36 | 100% | 0 (1,526 steps) | 99.68% |

Drift counts slips of the agent's map position against RAM within one map segment and one RAM map; it never slipped. Battle-state agreement leaves out fade frames, which read "transition" while RAM is already in battle. A lost battle sends the player back to Pallet, so random still reaches Viridian, only later. Greedy equals type-aware on this route: Charmander knows only SCRATCH and GROWL, and Route 1 has only Normal/Flying species, so type never changes the choice.

### 5.2 Perception (test runs 15–19, unseen)

| Component | Metric | Target | Result |
|---|---|---|---|
| Screen state, CNN | accuracy (16,432 frames) | ≥ 99% | **99.49%**, 1.50 ms/frame |
| Screen state, heuristic | accuracy | — | 95.36%, 0.52 ms/frame (misses battle frames without the HUD) |
| Tiles, lookup | walkable/blocked vs RAM collision (266,883 cells) | ≥ 98% | 100% of the 97.6% of cells found |
| Tiles, CNN | walkable/blocked vs RAM collision; class vs labels | ≥ 98% | **100%** of all cells (also the 2.4% the lookup misses); class 100% |
| Tiles, unseen area | CNN walkable vs RAM (43,221 cells north of Viridian) | — | **89.97%** (Route 2 100%, Forest gate 88.8%, Viridian Forest 83.6%); lookup finds only 37.0% |
| Map position | error vs RAM (all 20 runs, 12,212 frames) | 0 drift | **0 tiles** on every frame; same with CNN screen states |
| Registration | correct steps (10,770 pairs) | 100% | phase correlation 10,766; with direction prior **10,770** |
| NPC detection | precision / recall / facing | ≥ 95% | **100% / 97.5% / 99.1%** |
| Species | top-1 among 151 | ≥ 95% | **100%** (716/716) |
| HP bar | MAE | ≤ 2% | **0.63%** enemy, 0.47% own |
| OCR | character accuracy | ≥ 99.5% | 99.10% against RAM text; **99.99%** on text actually drawn |

OCR note: of 3,773 mismatches with RAM, 3,713 are tiles where the screen does not yet show the RAM text (a box drawn before its text, a sprite over text, a flash). Only 60 are real misreads.

### 5.3 Capture chain

Screen localisation (500 frames, corner error ≤ 1 px): level 0 100%, level 1 97.0%, level 2 44.6% (median 1.06 px), levels 3–4 fail (> 5 px on 90–100% of frames).

Shade accuracy after the full chain (200 frames):

| Level | True corners, no restoration | Found corners | Found corners + restoration |
|---|---|---|---|
| 0 | 100% | 100% | 100% |
| 1 | 88.4% | 87.4% | **98.6%** |
| 2 | 67.8% | 62.2% | 63.1% |
| 3 | 65.8% | 48.6% | 33.0% |
| 4 | 62.8% | 5.7% | 5.4% |

Restoration methods (each degradation type alone at strength 2, true corners): every 3×3 smoothing filter drops shade accuracy to 59–77%, because one screen pixel is only about 2 image pixels wide. Wiener with a blind σ is the only method that helps blur without hurting the other types (defocus 58.6% → 68.2%). Flattening the light fixes the illumination gradient (97.6% → 100%). Averaging 4 frames is best on Gaussian noise (35.5 dB) but needs a still screen. Histogram equalisation is the worst (28% on Gaussian noise). Full table: `runs/restore_eval.txt`.

### 5.4 Registration under degradation: phase correlation versus ORB

Correct steps out of 200 pairs, each frame through the capture chain:

| Level | Phase correlation | ORB + RANSAC | Direction prior (before fallback) | Direction prior + phase fallback |
|---|---|---|---|---|
| 0 | 100% | 70.5% | 100% | 100% |
| 1 | 99.5% | 72.0% | 97.0% | 99.5% |
| 2 | 81.5% | 69.0% | 23.5% | 82.0% |
| 3 | 9.0% | 4.5% | 1.0% | 10.0% |
| 4 | 0% | 0% | 0% | — |

ORB fails on about half the real steps even on clean frames. Grass and path tiles repeat every 16 px, so after a one-cell scroll keypoints match their own old position, and RANSAC's largest consensus is zero shift. A Lowe ratio test and masking the player sprite did not help. This sweep found the weakness of the direction prior at level 2 and led to the phase-correlation fallback.

### 5.5 Robustness sweep

![robustness](runs/robustness.png)

| Level | Restoration on | Restoration off |
|---|---|---|
| 0 | 3/3 | — |
| 1 | 3/3 (340 decisions, 2.3 map segments) | 2/3 (458 decisions, 5 segments) |
| 2 | 0/3 | 0/3 |
| 3 | 0/1 | 0/1 |
| 4 | 0/1 | 0/1 (screen not found on 79% of frames) |

At level 1 restoration keeps the agent close to its clean behaviour. Without it, registration slips more often (more map segments) and one seed fails. Level 2 was traced: blur erases the 1-px floor stripes of Oak's Lab, so the floor reads as flat white, and the screen-state CNN (trained on clean frames only) calls the room "transition" or "battle". The agent then waits. The hand-written heuristic still reads the same frames as overworld.

## 6. Bugs the evaluation found

- **NPC bumps.** When NPC detection stopped bumps from marking walls, an NPC that walked beside the player drew endless bumps, and the clean success rate fell from 10/10 to 8/10. Marking the cell "npc" at low confidence restored 10/10 with identical decisions per seed.
- **Registration at level 2.** The direction prior's difference cut-off rejected real steps on noisy frames and declared transitions. The phase-correlation fallback raised level-2 accuracy from 23.5% to 82.0% with no change on clean frames.
- **Misread move names.** OCR on a degraded frame read GROWL as `G  WL` and crashed the policy. Names now snap to the closest known move.

## 7. Limitations

- **Degradation beyond level 1.** The screen-state CNN and the tile lookup were trained on clean frames only; quantisation is limited by blur at level 2 (67.8% even with true corners, and Wiener with the true PSF does not recover it).
- **Stuck detection under noise.** Outside the overworld the watchdog compares raw frames, which never repeat under fresh noise, so it cannot fire in sim mode.
- **NPC in tall grass.** All 9 test misses are one NPC whose lower half is hidden by grass (NCC 0.74 against the 0.8 threshold).
- **Blackouts** warp the player to a place not yet seen, which starts a map segment that is never merged with the old one.
- **One route.** Only Pallet Town → Viridian City is evaluated; Route 1 has only Pidgey and Rattata, so the type-aware policy is never tested against a type advantage.
- **Tile labels and unseen tiles.** The 101 cell labels were checked from images by Claude at the user's request, not by hand in the labelling tool (3 of the RAM-based guesses changed). The tile CNN learns from those 101 cells only. In Viridian Forest, 95% of its walkable errors (4,109 of 4,336) are one unseen cell, the dark diagonal edge of the tree canopy, read as a door. The agent uses the lookup; the CNN is evaluated, not wired in.

## 8. Future work

- Train the screen-state and tile CNNs on degraded frames as augmentation, and measure the robustness curve again.
- Key the stuck watchdog on screen state and map position rather than raw frames.
- Ablations planned but not run: the map without blocked-move self-correction, and the planner without grass costs.
- Viridian Forest (trainer battles, line of sight) as a second goal; it was cut for time.

## 9. Reproducing

```
python -m pixeltrainer.run --headless --record runs/demo.mp4      # one episode, demo video
python evaluation/run_eval.py --out runs/eval/final                # 20 seeds × 3 policies
python evaluation/report.py runs/eval/final                         # tables into runs/eval/final/report/
bash tools/sweep_queue.sh                                       # component evals + robustness sweep
python evaluation/plot_robustness.py
python -m pytest
```
