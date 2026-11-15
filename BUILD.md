# PixelTrainer — Build Plan

Ordered task list for building the project described in [PLAN.md](PLAN.md). Work top to bottom. Each task has a "done when" check, and a task is not done until its check passes.

Legend: `[ ]` todo, `[x]` done, `(R)` needs the ROM running, `(H)` needs a human at the keyboard.

---

## Phase 0 — Setup and verification

- [x] **0.1 Repo setup.** git repo, `.venv` (system site-packages, reuses torch), `requirements.txt`, `pyproject.toml`, `.gitignore` (ROMs, states, data, runs).
- [x] **0.2 RAM boundary test.** `tests/test_ram_boundary.py` fails if `pixeltrainer/` imports `evaluation` or uses PyBoy state APIs (`memory`, `game_area*`, `get_sprite`, `tilemap_*`, …).
- [x] **0.3 ROM check.** `tools/check_rom.py`: title `POKEMON RED`, boots, writes `runs/boot.png`. Hash is a patched `[S][BF]` dump, not the clean one.
- [x] **0.4 Symbol file.** `roms/pokered.sym` from pret/pokered `symbols` branch. `PyBoy(rom, symbols="roms/pokered.sym")` then `symbol_lookup("wCurMap")` returns `(0, 0xD35E)`.
- [x] **0.5 Manual player.** `tools/play.py` (R)(H): windowed PyBoy (3x), PyBoy default keys in the game window. Commands typed in the terminal instead of F-keys (PyBoy's SDL window owns the keyboard): `save <name>`, `load <name>`, `truth`, `frame [name]`, `quit`. Used by hand to make the start states.
- [x] **0.6 RAM truth reader.** `evaluation/truth.py` (`read`, `sprites`, `tilemap`). Checked at 5 moments, see `runs/truth_check.md`. Fix: no map coords for the player's sprite slot.
- [x] **0.7 Start states (H).** `states/pallet_start.state` (Pallet, outside Oak's lab, Charmander L6) and `states/viridian_north.state` (Viridian, outside the Poké Center).
- [x] **0.8 Verify screen geometry.** `tools/probe_geometry.py`: grid aligned at (0,0), player cell (4,4) drawn 4 px up, 16 frames and 16 px per step, blocked step 0 px, seamless outdoor border, doors fade through black (~70 frames). Results in PLAN.md §3.2.

**Gate:** PLAN.md has no Phase 0 `[verify]` tags left. Passed 2026-11-14 (remaining tags belong to Phases 3, 5, 6).

---

## Phase 1 — Emulator harness and dataset

- [x] **1.1 `pixeltrainer/emu.py`.** `Emu(rom, state, headless=True, seed=0)`, `reset()` (load state + random idle 0–300 frames from seed), `frame() -> np.ndarray (144,160) uint8 grey`, `press(button)`, `wait_until_stable(max_frames)`, `step(button) -> frame`. Exposes the `PyBoy` object only through `self._pyboy` for the evaluator. Done when a test shows two runs with the same seed produce identical frames, and different seeds diverge.
- [x] **1.2 Grey palette.** Map the RGBA screen to 4 shade indices 0–3 once in `frame()`. Every module works on shade indices. Done when `np.unique(frame())` ⊆ {0,1,2,3} on 100 random frames.
- [x] **1.3 `tools/record_dataset.py`.** Runs a scripted walker (random walk with goal bias, presses A in dialogue, picks move 1 in battles) and saves per stable frame: `frame.png` + `truth.json`. Output `data/raw/<run_id>/`. Run IDs 0–9 train, 10–14 validation, 15–19 test. Done when ~20k frames exist covering Pallet, Route 1, Viridian, buildings, battles. Done 2026-11-14: 20 runs, 65k frames; Pallet, Route 1, Viridian, player's house (maps 37/38). Re-recorded with overworld A presses (5%) to get dialogue. No Viridian buildings or Oak's lab.
- [ ] **1.4 Dataset viewer.** `tools/view_dataset.py`: step through frames with truth printed. Used to spot-check labels.

---

## Phase 2 — Screen state, grid, tiles

- [x] **2.1 Screen-state labels.** Derive overworld/battle/dialogue/menu/transition from truth fields (rules found in 0.6). Spot-check 200 frames by hand. Done when spot-check disagreement < 1%. Done: `evaluation/screen_labels.py` (RAM + glyph visibility, `tools/build_glyphs.py`). Spot-check 0/200 disagreements after fixing stale-menu frames and the in-battle party menu; 20/20 dialogue samples correct. Counts: overworld 30403, battle 26214, menu 6237, transition 1919, dialogue 462.
- [x] **2.2 Heuristic baseline.** `perception/screen_state.py::heuristic()`: text-box border pattern + battle HP-bar pattern + mean intensity for fades. Report its accuracy. Done: `perception/screen_state.py::heuristic` (8x8 glyph matches: HP-bar start, box corner, font in the top rows). Test runs 15-19: 95.36%, 0.52 ms/frame; misses battle frames without the HUD (intro, swirl, faint), 10.5% of battle. `evaluation/eval_screen_state.py`, `runs/screen_state_heuristic.txt`.
- [x] **2.3 Screen-state CNN.** Train (`tools/train_screen_state.py`), save `models/screen_state.pt`. Done when test accuracy ≥ 99% and inference < 2 ms per frame on CPU. Done: 4 strided conv layers + 3x3 pooling (`perception/screen_state.py::cnn`), `tools/train_screen_state.py`, 8 epochs on CPU. Test runs 15-19: 99.49%, 1.50 ms/frame (heuristic 95.36%); battle recall 0.990 vs 0.895. `runs/screen_state_cnn.txt`.
- [x] **2.4 `perception/grid.py`.** Frame → 9×10×16×16 cells, player cell index, cell hashing. Unit test with a synthetic frame. Done (`tests/test_grid.py`).
- [x] **2.5 Unique-cell inventory.** Collect all unique overworld cells from train runs with RAM walkable flag. Report count. Done: `tools/cell_inventory.py` -> `data/cells.npz`. 101 unique background cells from 6119 frames (sprite cells masked; ~30 are water animation frames); 17 walkable; RAM walkability never disagrees within a cell.
- [ ] **2.6 `tools/label_cells.py` (H).** Shows each unique cell enlarged with neighbours; keys for each class; walkable pre-filled. Saves `assets/cell_labels.json`. Done when every unique cell is labelled.
- [x] **2.7 Lookup classifier.** Exact-hash lookup, `unknown` on miss. Done: `perception/tiles.py::lookup`. Uses `assets/cell_labels.json` once 2.6 is done; until then the RAM-based guesses in `assets/cell_labels_auto.json` (`tools/label_cells.py --auto`). Test runs 15-19: 97.6% of visible cells found, walkable/blocked agrees with RAM collision on 100% of them.
- [ ] **2.8 Tile CNN.** Train on labelled cells (with shift ±1 px and noise augmentation). Evaluate on test runs (seen area) and on Viridian Forest frames (unseen area). Done when walkable accuracy ≥ 98% in the seen area.

---

## Phase 3 — Registration and map

- [x] **3.1 `mapping/registration.py`.** `shift(prev, cur, mask) -> (dx, dy, confidence)` via `cv2.phaseCorrelate`; snap to {0, ±16}. `is_transition(prev, cur)`. Test on recorded step pairs: shift matches RAM coordinate change on 100% of clean steps. Done: phase correlation alone 10766/10770 (the 4 misses are 16 px-periodic grass/path read as 0); with the pressed direction as a prior (MAD over the overlap for 0/1/2 cells) 10770/10770 incl. 57 ledge jumps and 4406 blocked. Masking NPCs with zeros hurts (98.2%), so only the player is masked. Seamless borders 213/213; all 26 door crossings are caught by the screen state (fade = transition) before registration. `evaluation/eval_registration.py`, `runs/registration_eval.txt`, `tests/test_registration.py`.
- [x] **3.2 `mapping/worldmap.py`.** Segments, global grid, position estimate, observation merge rule, blocked-move wall marking, door links between segments. Render to an image for debugging. Done: after a gap (battle, menu, dialogue) the frame is registered against the last overworld frame; if it does not register it is a new place, relocalised against all segments by exact cell keys (5 rarest seen keys as anchors, >= 80% of >= 30 known cells must match) or a new segment. Merge rule: replace a label when strictly more confident or the old one is < 0.8. Known limit: a blackout warps to a place not yet seen, which starts a segment that is never merged with the old one. `tests/test_worldmap.py`.
- [x] **3.2b World-map mosaic.** Paste every registered stable frame into one pixel image per segment (median over overlaps). Save `runs/<run>/mosaic.png`; shown in the live view. Done when the mosaic of a Pallet → Viridian walk has no visible seams. Done: per-cell shade counts, so the median is exact; moving NPCs drop out. `WorldMap.mosaic()`/`render()` (walls red, doors green, player blue); run 10 (Pallet → Viridian) has no seams: `runs/worldmap/10/mosaic_0.png`. Live view still to do (7.x).
- [x] **3.3 Map replay test.** Replay a recorded Pallet → Viridian walk through registration + map. Done when position error vs RAM is 0 tiles on every step (after per-segment origin alignment), and the rendered map looks right. Done: `evaluation/eval_worldmap.py`, all 20 runs: 12212/12212 overworld frames at 0 tiles error, no segment mix-ups (outdoor map offsets from RAM border crossings). With the screen-state CNN instead of RAM labels: 12262/12262. Every extra segment follows a blackout. `runs/worldmap_eval.txt`, `runs/worldmap_eval_cnn.txt`.

---

## Phase 4 — Planner and overworld agent

- [x] **4.1 `planning/astar.py`.** Grid A*, costs from PLAN.md §4.6, ledges as directed edges. Unit tests: wall detour, grass avoidance, ledge one-way. Done: `tests/test_astar.py`.
- [x] **4.2 `planning/explore.py`.** Goal = compass direction + landmark template. Frontier scoring; frontier limited to the region grown from the player cell (region growing over passable cells). Unit test on a synthetic partial map. Done: frontier = unknown cells next to the grown region (ledges one-way); score = progress - 0.5 x steps; path by astar. Landmark is 4.3. `tests/test_explore.py`.
- [x] **4.3 Landmark templates.** Capture Pokémon Center roof (and Viridian sign if useful) into `assets/landmarks/`. Done: the Center's "POKé" sign (`tools/capture_landmark.py` -> `assets/landmarks/poke_sign.png`), NCC >= 0.9 (`perception/landmark.py`). Test runs 15-19: 237 Viridian frames above the threshold, best score off Viridian 0.44.
- [x] **4.4 `pixeltrainer/agent.py` (overworld only).** State machine; battles handled by a placeholder (press A, pick first move). Watchdog for stuck states. Done. A door entered by accident (`pallet_start` stands under Oak's lab) is left at once and labelled `door`; cells the player stood on count as walkable (0.5); with no frontier inside a building, back to the arrival cell and walk south.
- [x] **4.5 First end-to-end run.** Done when the agent reaches Viridian from `pallet_start` in ≥ 8 of 10 seeds (checked by the evaluator via RAM). Done: 10/10 seeds, 87-324 decisions (0-236 of them in battle). `evaluation/run_agent.py`, `runs/agent_eval.txt`. Tile labels are still the unchecked guesses (2.6).

**Milestone A: navigation demo works.** Reached: 10/10 seeds (4.5).

---

## Phase 5 — Battle perception and policy

- [x] **5.1 Font templates.** `tools/capture_templates.py font`: collect 8×8 glyphs from text frames, labelled via VRAM tilemap / known strings. Save `assets/font/`. Done: from the ROM's FontGraphics (`tools/build_font.py` -> `assets/font.npz`, 86 characters); matches all 74 text glyphs captured from our own frames.
- [x] **5.2 `perception/ocr.py`.** Read a text region into a string; cursor `▶` locator. Done when character accuracy ≥ 99.5% on test frames. Done: per 8x8 tile, exact match then nearest by Hamming distance (<= 6). `evaluation/eval_ocr.py`, test runs 15-19 (battle/menu/dialogue frames): characters 99.10% vs wTileMap; of the 3773 mismatches, 3713 are tiles where the screen differs from RAM (box drawn before its text, sprites over text, flashes) and 60 are real misreads, so drawn-text accuracy is 99.99%. `runs/ocr_eval.txt`, `tests/test_ocr.py`.
- [x] **5.3 Species templates.** Capture enemy front sprites from battles, labelled by RAM species. Also import all 151 from pret `gfx/pokemon/front/` for the full-gallery test. Done: all 151 from pret (`tools/import_species.py` -> `assets/species.npz`), placed as the game places them (bottom-aligned, centred in 56x56). Our own battle frames match them exactly (NCC 1.000), so no separate capture was needed.
- [x] **5.4 `perception/battle.py`.** Species ID by NCC and by a CNN embedding (compared). HP bar by Otsu threshold + filled-pixel count, HP bar fraction, battle menu state (main / fight / text). Done when species top-1 ≥ 95% and HP MAE ≤ 2% on test frames. Partly done: `perception/battle.py`, species by NCC against all 151, HP bars, own HP numbers by OCR, menu state and cursor. Test runs 15-19, 716 menu frames: species 716/716, both HP bars pixel-exact (MAE 0.63% / 0.47%, the bar itself rounds), own HP numbers 716/716, cursor 716/716. Only Pidgey and Rattata appear on Route 1. CNN-embedding comparison not done yet (more useful on degraded frames, Phase 8). `evaluation/eval_battle.py`, `runs/battle_eval.txt`.
- [x] **5.5 `policy/gamedata.py`.** Gen 1 type chart, move table, species types (only what the routes need, plus all types for the chart). Unit tests on known matchups (e.g. Ghost vs Psychic = 0 in Gen 1). Done: all 151 species, 165 moves and the type chart, generated from the ROM by `tools/gen_gamedata.py`; `tests/test_gamedata.py` checks them against the ROM.
- [x] **5.6 `policy/battle_policy.py`.** random, greedy, type-aware; each returns `(move, reason_text)`. Done: `tests/test_battle_policy.py` (super-effective beats power, Ghost immune to Normal, status only when nothing damages). Fixed-damage moves count as status (shortcut).
- [x] **5.7 Battle control in `agent.py`.** Navigate FIGHT → move with the cursor; handle level-up, move-learn prompts (decline new moves: keep it simple and deterministic), fainting. Done: main menu -> FIGHT, fight menu -> policy move (chosen once per visit) via the cursor, text -> A, move-learn declined. Directions pressed in battle no longer reach the map.
- [x] **5.8 Battle evaluation.** Win rate per policy over seeded wild battles. Done: `evaluation/run_agent.py --policy all`, 10 seeds each, Pallet -> Viridian: random 12/14 battles won (2 blackouts), greedy 15/15, type-aware 15/15 (67 decisions per win vs 92 for random); all 30 runs reach Viridian. Greedy = type-aware here: Charmander has only SCRATCH/GROWL and Route 1 has only Normal/Flying species, so types never change the choice. `runs/agent_eval_<policy>.txt`.

---

## Phase 6 — NPC detection and stretch goal (6.1–6.2 core; 6.3–6.4 optional)

- [ ] **6.1 NPC templates.** Capture NPC sprites in 4 facings, labelled from the RAM sprite table.
- [ ] **6.2 `perception/npc.py`.** Candidates by connected components on non-background shades (8-connectivity), scored by template NCC, duplicates removed by overlap. Outputs cell, box, facing. A bump next to a detected NPC is not written into the map as a wall. Done when precision and recall ≥ 95% (IoU ≥ 0.5).
- [ ] **6.3 Sight-line costs** in the planner; trainer flag per sprite type.
- [ ] **6.4 Viridian Forest runs** from `viridian_north.state`.

---

## Phase 8 — Capture front-end and robustness (PLAN.md §6A, simulated only)

- [x] **8.1 `evaluation/degrade.py`.** Seeded degradations: geometric, blur, noise models (Gaussian, salt-and-pepper, Poisson), photometric, moiré. Each type at strengths 1–4, plus combined levels 0–4. It gets the clean frame; the agent gets only its output. Done when a contact sheet of every type × strength is saved. Done: `runs/degrade/contact.png`, `tests/test_degrade.py`. `meta` returns the true corners and PSF.
- [ ] **8.2 `capture/localise.py`.** Canny + contour quad fit, Hough-line fallback, Harris corner refinement. Done when corner error ≤ 1 px at level 2 on 500 frames. In progress: background-model subtraction -> contour quad -> gradient edge passes -> step-model edge passes (half-height crossing, Huber line fits). 500 test frames (`evaluation/eval_localise.py`, `runs/localise_eval.txt`): level 0 94.4% <= 1 px, none fail; level 1 66.8% (20.8% > 5 px); level 2 37.8% (45.4% > 5 px). Main failure: pale clutter boxes touching the screen join the coarse mask. Tried and dropped: self-template ECC (an offset is a fixed point), ECC pre-smoothing (~1 px bias).
- [ ] **8.3 `capture/rectify.py`.** Perspective transform to 160×144. Done when rectified level-1 frames quantise to the clean frame at ≥ 99% of pixels.
- [ ] **8.4 `capture/restore.py`.** Mean, Gaussian, median, adaptive median, Laplacian/unsharp sharpening, Gaussian/Butterworth low-pass, FFT notch, Wiener, gamma, histogram equalisation/CLAHE, N-frame averaging; Haar wavelet optional. Done when the PSNR/SSIM table per method × degradation type exists.
- [ ] **8.5 `capture/quantise.py`.** Grey, then multi-Otsu to 4 shades. Done when shade accuracy ≥ 99% at level 2.
- [ ] **8.6 Registration comparison.** Phase correlation vs ORB + descriptor matching + RANSAC, across levels.
- [ ] **8.7 Robustness sweep.** `run_eval.py --input sim:<level>` for levels 0–4, restoration on/off. Produces the robustness curve.

---

## Phase 7 — Live view, evaluation, report

- [ ] **7.1 `pixeltrainer/viz.py`.** Side-by-side window + mp4 writer.
- [ ] **7.2 `pixeltrainer/run.py`.** CLI: `--state`, `--goal`, `--policy`, `--seed`, `--headless`, `--record`.
- [ ] **7.3 `evaluation/run_eval.py`.** N seeds × goals × policies; logs per-frame perception and truth to `runs/eval/<timestamp>/`.
- [ ] **7.4 `evaluation/report.py`.** All tables and plots from PLAN.md §5, plus ablations, into `runs/eval/<timestamp>/report/`.
- [ ] **7.5 Demo video.** Record one clean run.
- [ ] **7.6 Written report.** Method, the RAM-use statement, results, limitations, future work.

---

## Order of attack and cut-lines

1. Phases 0–4 first, to reach Milestone A (navigation demo).
2. Then Phase 5, Phase 6.1–6.2, Phase 8, then Phase 7.
3. Phase 6.3–6.4 only if time remains.

If time gets tight, cut in this order:
1. Phase 6.3–6.4 (stretch goals).
2. Wavelet denoising and Wiener in 8.4.
3. The species CNN in 5.4 (keep NCC).
4. The tile CNN's unseen-area test.
5. Ablations.

Never cut: the RAM boundary test, the evaluation report, the demo video, the robustness curve (8.7).
