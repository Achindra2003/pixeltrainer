<div class="cover">
<h1 class="title">User Manual</h1>
<p class="sub">Computer Vision (MCA509D-5) — CIA 3</p>
<p class="sub">PixelTrainer — A Vision-Only Agent that Plays Pokémon Red</p>
<p>Achindra Sharma — 2547105 (Individual Submission)</p>
<p>4MCA A</p>
<p>Repository: <a href="https://github.com/Achindra2003/pixeltrainer">github.com/Achindra2003/pixeltrainer</a></p>
</div>

<div class="contents">
<h2 class="nobreak">Contents</h2>

- Background: the game and the task
- At a glance
- 1\. Installation
- 2\. Running the program
- 3\. Explanation of the code
    - 3.1 How the project is organised
    - 3.2 What happens in one decision
    - 3.3 How the course topics fit in
    - 3.4 How I checked the code
- 4\. The input image and why it is necessary
- 5\. Output images, with justification
- 6\. Troubleshooting
- Appendix A. The algorithms in detail, and why I chose them
</div>

## Background: the game and the task

### What Pokémon Red is

Pokémon Red is a role-playing game made by Game Freak and Nintendo for the original Nintendo Game Boy. It came out in 1996 in Japan and in 1998 internationally. The Game Boy has a small 160×144 pixel screen that shows only 4 shades of grey, with no colour. The game is played with a direction pad and two buttons, A (confirm) and B (cancel).

### How the game is played

The player controls a young trainer, seen from above, who walks around a world made of square tiles: towns, paths, trees, fences, houses and patches of tall grass. The game switches between a few kinds of screens:

- **The map (overworld).** The player walks one tile at a time. The camera follows them, so the player always stays in the middle of the screen while the world scrolls past.
- **Battles.** Walking through tall grass can start a fight with a wild Pokémon (a creature). The screen changes to show the enemy, both sides' health (HP) bars and a menu. Battles are turn-based: each turn the player picks one of their Pokémon's moves, and some moves work better than others depending on the creatures' "types" (for example, Water beats Fire).
- **Text boxes and menus.** People and signs talk to the player through text boxes, which are cleared by pressing A.

### What my program has to do

The task is the first journey of the game. The player starts in their home town, **Pallet Town**, walks north along **Route 1** (a path full of tall grass, ledges and other people), wins any wild battles on the way, and arrives in **Viridian City**, which is marked by its Poké Center sign. A human does this by looking at the screen, recognising what they see, remembering where they've been and deciding where to go. My program has to do exactly the same thing, using only the screen image.

### Why this makes a good computer vision problem

Each skill a human player uses maps onto a computer vision task:
- telling a battle from the map is **image classification**
- seeing walls and paths is **segmentation** of the scene into regions
- spotting people and recognising the enemy creature is **detection** and **template matching**
- reading menus is **OCR**
- knowing where you are as the screen scrolls is **image registration**

On top of that, a photo of a real Game Boy screen would be blurred, tilted and noisy, which brings in **image restoration**. So the project uses the full course, from basic pixels and quantisation up to features, segmentation and neural networks, inside one working system. You can see examples of each screen type in Figure 2 (Section 5.1).

## At a glance

::: lead
**PixelTrainer is a program that plays Pokémon Red by looking at the screen, the way a person does.** It starts in Pallet Town, walks north through Route 1, fights the wild Pokémon it meets, and stops when it recognises the Poké Center sign in Viridian City. It never reads the game's memory: **every decision comes from computer vision on the screen image.**
:::

![Figure A — How PixelTrainer works: one screen image goes in, a chain of computer-vision stages reads it, and one button press comes out. Each box shows the question it answers, the method it uses and the syllabus unit it comes from.](docs/figures/fig0_pipeline.png)

### What makes this project different

::: highlight

- **It works from pixels only.** Most game bots read the game's memory to find out where the player is or how much HP an enemy has. Mine has to *see* all of that, and an automatic test fails if any agent code touches the memory.
- **It's a whole system, not a single detector.** About ten vision stages run together, live, on every frame (Figure A): screen classification, tile classification, person detection, position tracking, map building, path planning, sprite recognition, OCR, HP reading and goal detection.
- **It takes a real photo of the screen as well.** In camera mode the input is a tilted, blurred, noisy photo. The program finds the screen, restores it, straightens it and turns it back into a clean frame before playing. This is where most of the Unit 2 restoration topics are used.
- **It covers the syllabus.** 22 of the 23 topics taught so far are used in the code (Section 3.3). For each one I chose the specific algorithm that suited the problem, and Appendix A explains why.
- **Every result is measured.** All the accuracy numbers come from recorded runs the models never saw during training, and they are checked against the game's true state.

:::

### Results

| What was measured | Result |
|---|---|
| Reached Viridian City (20 seeds × 3 battle strategies) | **60 / 60 episodes** |
| Screen type (16,432 unseen frames) | 99.49% correct |
| Walkable / blocked tiles (266,883 unseen cells) | 100% correct |
| People on the map | 100% precision, 97.5% recall |
| Position tracking (12,212 frames) | 0 tiles of error, so the map never drifted |
| Enemy species (716 test frames, 151 possible) | 100% correct |
| Camera mode, level 1 photo | 99.95% of pixels restored correctly; goal reached in 3/3 runs |

### How to read this manual

| If you want to… | Go to |
|---|---|
| know what Pokémon Red is and what the task is | Background |
| install it and run it | Sections 1–2 |
| understand the code, step by step | Section 3 |
| see why the input image matters | Section 4 |
| see the output images and why they are correct | Section 5 |
| see the exact algorithms and why I chose them | Appendix A |

---

## 1. Installation

You need Windows 10 or 11, Python 3.11 and your own copy of the Pokémon Red ROM. I can't include the ROM for copyright reasons.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy <your dump>.gb roms\pokered.gb
.\.venv\Scripts\python tools\check_rom.py      # checks the ROM
.\.venv\Scripts\python -m pytest -q            # 47 tests, all should pass
```

The two trained models are `models/screen_state.pt` and `models/tiles.pt`. They come with the submitted zip. On a fresh clone from GitHub they are missing, and you can retrain them with `python tools/train_screen_state.py` and `python tools/train_tiles.py`.

The agent starts from a save state, `states/pallet_start.state`. Save states are made from the ROM, so neither the zip nor GitHub includes one. To make it, run `python tools/play.py`, play until the player is standing in Pallet Town, then type `save pallet_start` in the terminal.

## 2. Running the program

Run every command from the project folder, using the virtual environment's Python (`.\.venv\Scripts\python`).

| What you want | Command | What you get |
|---|---|---|
| Watch the agent play | `python -m pixeltrainer.run` | A game window at normal speed |
| Record a demo video | `python -m pixeltrainer.run --headless --record runs/demo.mp4` | A video with the game on the left and the agent's map, plan and decisions on the right |
| Try a different run | `--seed 3 --policy greedy` (policies: `random`, `greedy`, `type`) | A different episode, because the wild encounters change with the seed |
| Regenerate the figures | `python tools/make_figures.py` | `docs/figures/fig1…fig9.png` |
| Full evaluation | `python evaluation/run_eval.py --out runs/eval/final`, then `python evaluation/report.py runs/eval/final` | Result tables for 20 seeds × 3 policies |
| Camera-style input | `python evaluation/run_eval.py --input sim:1 --seeds 0-2` | The same agent playing from degraded "photos" of the screen |

When an episode ends, the console prints a one-line summary, for example:

```
landmark seen after 133 decisions, 2 map segments, watchdog {}
```

This means the agent reached the goal after pressing 133 buttons. It built 2 separate pieces of map on the way, and it never got stuck (the watchdog never had to step in). If the goal is not reached in time, the line says `budget used up` instead.

## 3. Explanation of the code

### 3.1 How the project is organised

I split the code by job. Everything the agent uses lives in `pixeltrainer/`. Anything that is allowed to look at the game's real memory, for scoring and labelling, is kept separate in `evaluation/`.

```
pixeltrainer/            the agent: sees pixels only
  emu.py                 emulator wrapper: frame in (4 shades), button out
  agent.py               decision loop (state machine)
  run.py, viz.py         command line, demo video
  perception/            what is on the screen
    screen_state.py      which of 5 screen types (CNN)
    grid.py, tiles.py    16x16 cells, walkable or blocked (lookup + CNN)
    npc.py               people on the map (connected components + NCC)
    battle.py, ocr.py    species, HP bars, menus, text
    landmark.py          goal sign (template matching)
  mapping/
    registration.py      how far the screen moved (phase correlation)
    worldmap.py          the map built from registered frames
  planning/
    explore.py           where to go next (region growing, frontier)
    astar.py             how to get there (A* search)
  policy/battle_policy.py  which attack to use
  capture/               camera mode: find, restore, straighten the screen
    localise.py, restore.py, rectify.py, quantise.py
evaluation/              scoring against the game's true state (the only code that reads RAM)
tools/                   training, labelling and figure scripts
tests/                   47 unit tests; test_ram_boundary.py fails if the agent reads RAM
```

### 3.2 What happens in one decision

The main loop is `run()` in `pixeltrainer/agent.py`. Each time round the loop, the agent looks at one frame and presses one button. Figure A shows the whole chain. Below, each stage is described in three parts: the **problem** it solves, the **idea** behind the solution, and the **result** on unseen data. The exact algorithms are in Appendix A.

::: stage
#### <span class="num">1</span> Read the screen <span class="files">`emu.py`</span>

[Problem]{.lbl} The agent needs the screen in a form it can compute with.

[Idea]{.lbl} The Game Boy has only 4 grey shades, so each pixel becomes a number from 0 (white) to 3 (black). In camera mode, the `capture/` module first turns the photo back into this form (step 10).
:::

::: stage
#### <span class="num">2</span> What kind of screen is this? <span class="files">`perception/screen_state.py`</span>

[Problem]{.lbl} Walking logic is useless in a battle, and battle logic is useless on the map, so the agent has to know which situation it's in first.

[Idea]{.lbl} A small CNN that I trained looks at the whole frame and labels it overworld, battle, dialogue, menu or transition. The layout gives it away: a battle has its display in the corners, while dialogue has a text box along the bottom.

[Result]{.lbl} 99.49% correct, against 95.36% for a hand-made rule-based version I wrote for comparison.
:::

::: stage
#### <span class="num">3</span> Where can I walk? <span class="files">`perception/grid.py`, `tiles.py`</span>

[Problem]{.lbl} The agent needs a map of walkable and blocked cells to plan a route.

[Idea]{.lbl} The game world is built from 16×16 tiles, so I cut the frame into a 9×10 grid. Each cell gets a fingerprint (a hash of its pixels), which is looked up in a table of tiles I labelled. A small CNN handles cells the table has never seen.

[Result]{.lbl} 100% agreement with the game's own collision data.
:::

::: stage
#### <span class="num">4</span> Is anyone in the way? <span class="files">`perception/npc.py`</span>

[Problem]{.lbl} People walk around, so they can't be treated as permanent walls.

[Idea]{.lbl} Character sprites have black outlines. I group touching black pixels to find candidate boxes, then confirm each one by matching it against the character templates (normalised cross-correlation). The best match also tells me which way the person is facing.

[Result]{.lbl} 100% precision and 97.5% recall.
:::

::: stage
#### <span class="num">5</span> Where am I? <span class="files">`mapping/registration.py`, `worldmap.py`</span>

[Problem]{.lbl} Without reading memory, the agent doesn't know its own position.

[Idea]{.lbl} Every step scrolls the screen by exactly 16 pixels, so comparing two frames shows how far the agent moved. For noisy frames, phase correlation (based on the Fourier transform) finds the shift instead. Each frame is then pasted onto a growing world map at that position.

[Result]{.lbl} Right on all 10,770 test steps, so the map never drifted (0 tiles of error).
:::

::: stage
#### <span class="num">6</span> Have I arrived? <span class="files">`perception/landmark.py`</span>

[Problem]{.lbl} The agent needs to know when to stop.

[Idea]{.lbl} Template matching against the Poké Center sign. A match score of 0.9 or more means the goal is reached.

[Result]{.lbl} Every one of the 60 evaluation episodes ended here.
:::

::: stage
#### <span class="num">7</span> Where do I go next? <span class="files">`planning/explore.py`, `astar.py`</span>

[Problem]{.lbl} The agent can only see a small part of the world at a time, so it has to explore.

[Idea]{.lbl} Region growing from the player's cell marks everywhere it can reach. The unknown cells along the edge (the "frontier") are where new map gets revealed. The agent picks the frontier cell that gets it furthest north without too long a detour, and A\* search finds the path there.

[Result]{.lbl} The seed-0 run reached Viridian in 133 button presses.
:::

::: stage
#### <span class="num">8</span> Battles: what am I fighting, and which move should I use? <span class="files">`perception/battle.py`, `ocr.py`, `policy/battle_policy.py`</span>

[Problem]{.lbl} To fight well, the agent has to read the battle screen just as a player would.

[Idea]{.lbl} The enemy is identified by matching its sprite against all 151 Pokémon. The menus and move names are read with OCR, by matching each 8×8 letter against the game's font. The HP bars are read by counting filled pixels. The agent then picks the move with the highest expected damage.

[Result]{.lbl} Species 100% correct, OCR 99.99%, HP bars within 0.63% on average.
:::

::: stage
#### <span class="num">9</span> Getting unstuck

If nothing on screen changes for 8 decisions, a watchdog presses a random button. It's a safety net for the rare cases where the agent gets stuck.
:::

::: stage
#### <span class="num">10</span> Camera mode: turning a photo back into a clean frame <span class="files">`capture/`</span>

[Problem]{.lbl} A photo of the screen is tilted, blurred, unevenly lit and noisy, so none of the exact methods above work on it directly.

[Idea]{.lbl} Four steps from the syllabus:
1. Find the screen, using background subtraction, Sobel edges and Hough lines.
2. Restore it, by evening out the lighting and then applying Wiener deblurring.
3. Straighten it, with a perspective warp.
4. Reduce it back to 4 shades, by splitting the histogram at its valleys.


[Result]{.lbl} On a level-1 photo, 99.95% of pixels come back correct and the agent still reaches the goal. Level 2 is where the method reaches its limit (Section 5.8).
:::

### 3.3 How the course topics fit in

When I planned the project, I tried to use the topics we covered in class wherever they actually solved a problem, rather than adding them just to tick boxes. The table below goes through every topic taught so far (Units 1 and 2, and Unit 3 Topic 1) and shows where it appears. Rows marked **(agent)** run while the agent is playing. The others are in the camera-mode pipeline, or in experiments I ran to choose the best method.

| Syllabus topic | Where in the code | How I used it |
|---|---|---|
| **U1-T1** Simple image formation model | `evaluation/degrade.py`, `capture/restore.py` | I model a camera photo as g = h ∗ f + η (blur h, noise η). Uneven lighting multiplies the image (illumination × reflectance), so `flatten_light` divides it back out. |
| U1-T1 Types of digital images | `emu.py`, `capture/` | The emulator's colour screen becomes greyscale, then a 4-level indexed image **(agent)**. Binary masks mark the screen and the dark pixels. |
| U1-T1 Correlation and convolution | `perception/npc.py`, `battle.py`, `ocr.py`, `landmark.py`; `capture/restore.py`; CNNs | Template matching by normalised cross-correlation **(agent)**. Convolution in every spatial filter and inside the CNN layers **(agent)**. |
| U1-T1 Fundamental steps of image processing | the whole pipeline (Section 3.2) | Acquisition (emulator or photo), preprocessing (restoration), segmentation (tiles, people), recognition (species, text, sign), then interpretation (the decision). |
| **U1-T2** Sampling, quantisation and resolution | `emu.py`, `capture/quantise.py`, `capture/rectify.py` | The 4 grey levels become shade indices **(agent)**. A photo is re-sampled back to the 160×144 screen resolution. |
| U1-T2 Neighbours and connectivity | `perception/npc.py`, `perception/grid.py`, `planning/explore.py` | 8-connected components find dark sprite outlines. 4-neighbour moves are used on the grid and in path planning **(agent)**. |
| U1-T2 Distance measures | `planning/astar.py`, evaluation | Manhattan (city-block) distance guides A\* **(agent)**. Euclidean distance measures how far off the detected corners are. |
| U1-T2 Colour models | `capture/localise.py`, `restore.py` | Colour to greyscale conversion. The colour relative to the background tells the grey screen apart from coloured clutter around it. |
| U1-T2 Geometric transformations | `mapping/worldmap.py`, `capture/rectify.py` | Translation places each frame on the map **(agent)**. A perspective warp straightens a tilted photo. |
| **U2-T1** Grey-level transforms and point operations | `capture/restore.py`, `quantise.py` | Gamma correction. Multi-level thresholding back into 4 shades. |
| U2-T1 Histogram processing and equalisation | `capture/quantise.py`, `restore.py` | The 4 shades are separated at the valleys of the histogram. Histogram equalisation and CLAHE were tested too. |
| U2-T1 Degradation, restoration and noise models | `evaluation/degrade.py` | Gaussian, salt-and-pepper and Poisson noise; defocus and motion blur; gamma; uneven light; colour cast; moiré. Four levels of strength. |
| U2-T1 Restoration in the presence of noise | `capture/restore.py` | Median and adaptive median filters, and averaging several frames. |
| U2-T1 Smoothing and sharpening spatial filters | `capture/restore.py` | Mean, Gaussian and median smoothing; unsharp masking and Laplacian sharpening. |
| **U2-T2** Fourier transform | `mapping/registration.py` | Phase correlation measures how far the screen scrolled **(agent fallback)**. |
| U2-T2 Frequency-domain filters | `capture/restore.py` | Gaussian and Butterworth low-pass filters, a notch filter for moiré, and Wiener deconvolution (used in the camera pipeline). |
| U2-T2 Wavelets | `capture/restore.py` (`wavelet_denoise`) | A 2-level Haar wavelet transform with soft thresholding of the detail bands (BayesShrink) to remove noise. |
| **U3-T1** Local and global features | `perception/tiles.py`, `screen_state.py`, `quantise.py` | Local: a single 16×16 cell or one sprite. Global: the layout of the whole frame, and its histogram. |
| U3-T1 Feature detection, description and matching | `perception/tiles.py`, `npc.py`, `battle.py` | Each cell is described by a hash of its pixels and matched against a table **(agent)**. Sprites and letters are matched by NCC **(agent)**. |
| U3-T1 Point, line and edge detection | `capture/localise.py` | Sobel edges, Hough lines and line intersections find the four corners of the screen in a photo. |
| U3-T1 Thresholding | `capture/quantise.py`, `localise.py`; detection scores | Histogram-valley thresholds, a threshold on difference from the background, and match-score thresholds (0.8 for people, 0.9 for the sign) **(agent)**. |
| U3-T1 Region growing | `planning/explore.py` | Grows the reachable area out from the player's cell. Its border is the frontier **(agent)**. |
| U3-T1 Region splitting and merging | not used | I didn't need it. The game image already comes split into a fixed 16×16 grid, and region growing does the merging. |
| CNN (from my synopsis) | `perception/screen_state.py`, `tiles.py` | Two small CNNs classify the screen type and the tile types **(agent)**. |

To choose the restoration methods for camera mode, I tested every filter against every type of degradation in `evaluation/eval_restore.py`, measuring PSNR, SSIM and the share of pixels that end up with the right shade. The best combination was "flatten the lighting, then Wiener deconvolution", so that is what the agent uses. Wavelet denoising was interesting: it gave the best SSIM of any single-photo method on Gaussian noise (0.914 → 0.933) and Poisson noise (0.771 → 0.808). I still left it out of the chain, because quantisation already gets 99–100% of the shades right on those noise types, so it wouldn't change what the agent sees.

### 3.4 How I checked the code

Because the whole point of the project is "pixels only", I needed to be sure the agent never cheats. I kept the testing honest in three ways:

- Only the `evaluation/` code reads the game's memory. It uses it for two things: labelling training data offline (recorded runs 0–14), and checking the agent's answers afterwards.
- All the accuracy numbers come from runs 15–19, which were never used for training.
- `tests/test_ram_boundary.py` fails automatically if any agent module touches the emulator's memory.

The main results (full tables in `REPORT.md`):
- the agent reached Viridian City in **60 out of 60** episodes
- screen type correct 99.49% of the time
- tiles 100% correct
- people found with 100% precision and 97.5% recall
- enemy species 100% correct
- HP bars within 0.63% on average
- map position error of 0 tiles

---

## 4. The input image and why it is necessary

### 4.1 What the input is

![Figure 1 — The input: one 160×144 Game Boy frame, 4 grey shades, on a fixed 16×16 cell grid.](docs/figures/fig1_input.png)

The agent gets **one 160×144 Game Boy frame per decision, and nothing else**. The emulator draws the screen in four grey levels (255, 153, 85 and 0), and `Emu.frame()` maps them to shade numbers 0 to 3 (Figure 1, middle). The histogram on the right has exactly four bars and nothing in between. That tells us a clean frame has no noise at all: the same tile always produces exactly the same pixels.

Three properties of this image are what make the whole project possible:

1. **It sits on a fixed grid.** The game world is built from 16×16 tiles and the camera follows the player. So the frame is always a 9×10 grid of cells lined up at the top-left corner (red lines in Figure 1), and the player is always in the same cell, row 4, column 4 (blue box). This is why I can classify tiles, place people and track position one cell at a time.
2. **The shades are exact.** With only 4 shades and no noise, a cell's pixels can be used as a fingerprint and looked up directly. Template matches also score a perfect 1.00 when they are right (Figures 4, 7 and 8).
3. **Scrolling is exact.** One step moves the image by exactly 16 pixels, so comparing two frames tells the agent precisely how many cells it moved (Figure 5).

### 4.2 Why the image is necessary

**The image is the only information the agent has.** The game's memory knows where the player is, what the map looks like and how much HP the enemy has, but my rule for this project is that the agent may not read any of it (Section 3.4). Every fact it uses has to be measured from the screen:

- where it is, by comparing consecutive frames
- where it can walk, from the tile classes
- who is in the way, by detecting people
- what it is fighting and which moves it has, by sprite matching and OCR
- when to stop, by spotting the sign

Each part of the pipeline relies on a different part of the image. If that part were missing, the agent would lose a specific ability:

| Stage | Part of the image it needs | What goes wrong without it |
|---|---|---|
| Screen type | the whole frame (battle display, text box, black fade) | The agent wouldn't know whether to walk, fight or press A |
| Tiles | each 16×16 cell | No map of walkable cells, so no path |
| People | dark sprite pixels and their shapes | The agent would walk into people, or treat them as permanent walls |
| Position | two frames in a row | No idea where it is, so no map |
| Battle | the enemy sprite box (56×56), HP bar rows, text rows | Unknown enemy and unknown moves, so no sensible attack |
| Goal | the sign's pixels | The agent would never know it had arrived |

The frame also has to be **still**. Before each decision, the emulator keeps running until the screen hasn't changed for 8 frames and isn't blank (`Emu.wait_until_stable`). A frame caught halfway through a scroll would break the 16-pixel grid that everything depends on.

### 4.3 A second input: a simulated camera photo

![Figure 9 (input column) — the second input: a simulated camera photo of the screen.](docs/figures/fig9_capture.png)

A perfect screen capture doesn't need any restoration, so on its own it wouldn't give the restoration topics from Unit 2 anything to do. To make them do real work, I added a mode where the agent sees a **simulated photograph of the screen** instead (`--input sim:<level>`, made by `evaluation/degrade.py`). The photo follows the degradation model from class, `g = h * f + η`. The screen is tilted and placed on a cluttered background, blurred, lit unevenly, given a colour cast and a moiré pattern, and finally has noise added. Levels 1 to 4 make each effect stronger.

This input matters because it is the realistic case: someone pointing a phone camera at a real Game Boy. The neat tricks from Section 4.1 stop working on it straight away. The grid is tilted, there are no longer exactly 4 shades, and nothing repeats pixel for pixel. The photo has to go through four steps to become a clean frame again: find the screen, restore it, straighten it, and reduce it back to 4 shades (Section 5.8).

### 4.4 Good and bad inputs

| Input | What happens |
|---|---|
| Clean, still emulator frame | Works: 60/60 episodes |
| Simulated photo, level 1 (slight tilt, slight blur, noise, colour cast) | Works: 3/3 episodes with restoration |
| Simulated photo, level 2 and above (strong blur, gamma, uneven light, motion blur, moiré) | Fails: the blur wipes out 1-pixel details, so the shades can't be recovered (Figure 9, bottom row) |
| Frame caught during a scroll or fade | Never used: the program waits for a still frame |
| A different game, or the Game Boy Color palette | Not supported: the templates and tile labels are made for Pokémon Red's 4 shades |

---

## 5. Output images, with justification

All the figures here were generated by `python tools/make_figures.py`, either from a real run (seed 0) or from the unseen test runs. In each one the input is on the left and the output on the right. For each figure I first say what it shows, then explain why I'm confident the output is correct.

### 5.1 Screen type

![Figure 2 — Screen-state classification: one frame per class, with true label, CNN and heuristic output.](docs/figures/fig2_screen_state.png)

[What it shows]{.tag} One frame of each screen type, taken from test run 18. Above each frame are three labels: the true answer, the CNN's answer and the hand-made method's answer. All five are correct.

[Why it's right]{.tag}
- This stage has to come first, because every later stage only makes sense on one kind of screen. Looking for walkable tiles during a battle, or reading a move list while walking around, would just produce nonsense.
- The five types look very different: a full map, a battle display with sprites, a text box over the map, a text box over a menu, or an almost plain black or white fade. That's why a small CNN can separate them so reliably.
- On 16,432 frames it had never seen, the CNN was right 99.49% of the time. The hand-made method managed 95.36%. It mostly fails on the first frames of a battle, before the battle display has appeared.

### 5.2 Tile types

![Figure 3 — Tile classification per 16×16 cell: input frame, exact-hash lookup, CNN.](docs/figures/fig3_tiles.png)

[What it shows]{.tag} The Pallet Town starting frame, then the lookup method's labels (87 of the 90 cells recognised), then the CNN's labels (all 90 cells). Green is walkable, red is a wall (the house and fence), cyan is the sign and amber is the door. The player's cell is hatched in blue and ignored, because the sprite covers the ground underneath.

[Why it's right]{.tag}
- The labels match what you can see: the house and fence are blocked, and the path, grass edge and flower bed are walkable.
- The lookup leaves 3 cells grey (unknown): the house door, the player's cell and a person cut off at the top-left edge. A sprite on top of a cell changes its pixels, so the fingerprint no longer matches the table. The CNN still labels all of them, and it correctly picks out the door.
- On the unseen test runs (266,883 cells), both methods agreed with the game's own collision data on 100% of the cells they labelled. The lookup is exact but only recognises 97.6% of cells, while the CNN covers all of them.
- There is one visible mistake. The CNN colours the half-visible person at the top-left as a wall. The people detector needs to see the whole sprite, so it misses this one, and the CNN ends up looking at sprite pixels instead of ground. The cell really is blocked, though, because someone is standing there, so the planner isn't misled. It's a labelling limit, not a path error.
- This walkable/blocked grid is exactly what the planner searches, so getting it right matters: one wrong label would send the agent into a wall.

### 5.3 Finding people

![Figure 4 — NPC detection by template matching, with the detected facing direction.](docs/figures/fig4_npc.png)

[What it shows]{.tag} A test frame from Viridian City, with the detected person boxed in magenta. The label gives the direction they are facing (`left`) and the match score (1.00). The player in the middle isn't reported, because the agent always ignores its own sprite.

[Why it's right]{.tag}
- A person isn't a wall, because they move. So the planner marks their cell as only temporarily blocked: the agent walks around them and tries again later.
- A score of 1.00 means the box matches a template exactly. Anything above 0.8 counts as a detection.
- On the unseen test runs, detection had 100% precision, 97.5% recall and got the facing direction right 99.1% of the time. All 9 misses were the same person standing in tall grass with the lower half of the sprite hidden (score 0.74).

### 5.4 Working out position

![Figure 5 — Registration: the frame is located on the world map by its pixel shift.](docs/figures/fig5_registration.png)

[What it shows]{.tag} Two frames, before and after pressing **up**, followed by the phase-correlation result. It has one sharp peak at dx = 0, dy = +16 pixels (response 0.83), which means a move of (0, −1) cells. The last panel is the difference between the new frame and the old one shifted by 16 pixels.

[Why it's right]{.tag}
- The peak is exactly one cell (16 px), in the opposite direction to the walk. That's what should happen: when the player walks up, the scenery scrolls down.
- After lining up the two frames, the difference image is black almost everywhere. The only things left are the new row that scrolled in at the top, and the player sprite, which stays in the centre of the screen. Since everything else lines up, the shift must be right.
- Over 10,770 test steps, phase correlation alone was right 10,766 times, and with the pressed direction as a hint it was right all 10,770 times. That's why the map never drifted: compared with the game's memory, the position error was 0 tiles across 12,212 frames.

### 5.5 The map and the plan

![Figure 6 — The built world map and the A* plan to the next frontier.](docs/figures/fig6_map_plan.png)

[What it shows]{.tag} On the left is the demo-video view at decision 80: the game on one side, and on the other the agent's own map with its planned path in yellow and its target in magenta. On the right is the final map of the last segment. Walls are tinted red, doors green, and the player is boxed in blue.

[Why it's right]{.tag}
- The map was built purely by pasting frames at the positions registration gave them. Pallet Town, the fence line and Route 1 still join up correctly, with no seams or doubled tiles. That's visible proof that the position tracking works.
- The yellow path heads north through walkable cells and avoids every red cell, which is what the goal direction asks for.
- The agent made this trip in 133 decisions with 2 map segments, and the watchdog never had to step in. A new segment starts whenever a frame can't be placed on the existing map, for example after walking through a door into a building.

### 5.6 Reading a battle

![Figure 7 — Battle reading: species (NCC), HP bars, menu text (OCR) and the chosen action.](docs/figures/fig7_battle.png)

[What it shows]{.tag} On the main battle menu, the enemy is identified as **PIDGEY** (match score 1.00), both HP bars read 100%, and the cursor is on FIGHT, so the agent presses A. On the fight menu, OCR reads the moves `['SCRATCH', 'GROWL']`, and the agent prints its reasoning: "SCRATCH vs Pidgey: 1x -> exp 40".

[Why it's right]{.tag}
- Everything it read matches the screen: the name at the top says PIDGEY, both bars are full, and the move list shows SCRATCH and GROWL.
- SCRATCH is the sensible choice. GROWL does no damage at all, and SCRATCH (power 40, Normal type) does normal (1×) damage to a Normal/Flying Pidgey.
- On the unseen tests, the species was right 100% of the time (716/716), the HP bars were off by only 0.63% on average, and OCR read 99.99% of on-screen text correctly.

### 5.7 Reaching the goal

![Figure 8 — Goal reached: the Viridian City Poké Center sign is recognised.](docs/figures/fig8_landmark.png)

[What it shows]{.tag} The Poké Center sign template (left) and the last frame of the episode (right). The best match is boxed in green with a score of 1.000, which clears the 0.9 threshold, so the agent stops.

[Why it's right]{.tag}
- The box sits exactly on the "POKé" sign in the top-right corner.
- A score of 1.000 means a pixel-perfect match. I set the threshold at 0.9 so small differences are still accepted, but a different sign wouldn't be.
- In the final evaluation, all 60 episodes (20 seeds × 3 battle strategies) ended right here.

### 5.8 Camera mode: turning a photo back into a frame

![Figure 9 — Camera mode: photo → localise → rectify → quantise back to a 4-shade frame.](docs/figures/fig9_capture.png)

[What it shows]{.tag} Each row takes one simulated photo through the four steps:
1. **Find the screen.** The green outline comes from background subtraction, Hough lines and edge refinement.
2. **Restore it.** Even out the lighting, then apply Wiener deconvolution, with the amount of blur estimated automatically.
3. **Straighten it.** Warp the screen back to 160×144.
4. **Reduce it to 4 shades** by splitting the histogram at its valleys.

The last panel marks in white every pixel that still ended up with the wrong shade.

[Why it's right, and where it stops working]{.tag}

- **Level 1 (top row) works.** The corners are found to within 0.12 pixels. After restoration, 99.95% of pixels get the right shade, compared with 86.74% without it, so restoration fixes almost every error and only about a dozen isolated pixels stay wrong. Because the frame is clean again, the tile fingerprints and templates match as normal, and the agent reaches Viridian in 3 out of 3 runs (only 2 out of 3 without restoration).
- **Level 2 (bottom row) fails, and I've shown it on purpose.** The corners are still found to within 1.09 pixels, but the stronger blur wipes out the 1-pixel details of the tiles. Quantisation only recovers 63.24% of the shades, and the grass and walls disappear. Restoration can't bring back detail the blur has already destroyed; here it actually scores slightly worse than no restoration (68.75%). This is the real limit of my system: from level 2 upwards the agent can't play (Section 7 of `REPORT.md`). Including the failure shows where the method stops working, and why the level-1 result can be trusted.

### 5.9 Demo video

`runs/demo.mp4` is submitted alongside this manual. It shows a whole episode at 4 decisions per second, in the same layout as Figure 6 (left). You can watch the game, the agent's map filling in as it explores, its planned path, and a status line showing the decision number, screen type, button pressed, map segment and, during battles, the reason behind each move. I made it with `python -m pixeltrainer.run --headless --record runs/demo.mp4`.

---

## 6. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `FileNotFoundError: roms/pokered.gb` | The ROM is missing | Copy your own Pokémon Red dump to `roms/pokered.gb` |
| `FileNotFoundError: models/...pt` | The trained models are missing (they aren't stored on GitHub) | Use the models from the zip, or run `python tools/train_screen_state.py` and `python tools/train_tiles.py` (needs the recorded data in `data/raw/`) |
| `budget used up` | The episode went past 1,500 decisions | Raise the limit with `--max 3000`, or try another `--seed` |
| The agent stands still in camera mode | Level 2 or higher: the shades can't be recovered | Use `--input sim:1` or clean frames |
| `ModuleNotFoundError: evaluation` | The command was run from another folder | Run it from the project root |

---

## Appendix A. The algorithms in detail, and why I chose them

Most computer vision topics have several well-known algorithms. Feature description alone could mean SIFT, ORB, HOG or a CNN. A key part of this project was choosing the *right* one for each job, so this section explains exactly what my code does at each stage and why.

The guiding idea came from looking at the input (Section 4). A Game Boy screen is **exact**: it has only 4 shades, a fixed grid, and no noise or rotation. General-purpose methods like SIFT or YOLO are designed to cope with messy real-world photos, and that robustness isn't needed here. So for the clean screen I used the simplest method that is exactly right. I kept the heavier, more robust methods for camera mode, where the input really is messy.

### 1. Tile types: feature description and matching <span class="files">`perception/grid.py`, `tiles.py`</span>

- *Descriptor:* each 16×16 cell's 256 shade values are hashed (BLAKE2b, 8 bytes). The hash is the cell's fingerprint.
- *Matching:* a dictionary lookup in a table of 101 cells I labelled by hand. A cell that isn't in the table is marked "unknown".
- *Why not SIFT, ORB or HOG?* Those descriptors are built to survive rotation, scaling and lighting changes, and none of that happens on this screen. The same tile always produces the same 256 values, so an exact hash is both perfect and instant.
- *Backup CNN:* 3 convolution layers (16, 32 and 64 filters, 3×3) with batch normalisation and ReLU, followed by a fully connected layer for the 9 classes. I flatten the features instead of pooling them away, so the network keeps track of *where* a pattern sits in the cell (a ledge's edge is always at the bottom). I trained it with ±1 pixel shifts, so it still works on cells the table has never seen.

### 2. Screen type: a CNN <span class="files">`perception/screen_state.py`</span>

- 4 convolution layers with stride 2 (5×5, then 3×3 kernels), then average pooling down to a 3×3 grid, then a fully connected layer for the 5 classes.
- *Why pool to 3×3 instead of 1×1?* The layout is what tells the screens apart. A battle has its display in the corners, and dialogue has a text box along the bottom. A 3×3 grid keeps that rough layout.

### 3. Finding people: connected components + normalised cross-correlation <span class="files">`perception/npc.py`</span>

1. Take every black (shade 3) pixel, since all character sprites have black outlines.
2. Group touching black pixels into 8-connected components (`cv2.connectedComponentsWithStats`) and drop tiny specks.
3. Each sprite-sized box that touches a component becomes a candidate.
4. Score each candidate against every sprite template with **masked NCC**: NCC = Σ(a − ā)(b − b̄) / √(Σ(a − ā)² · Σ(b − b̄)²), counting only the pixels inside the sprite's outline, so the ground behind it doesn't matter.
5. A score of 0.8 or more is a person. The best-matching template also tells me which way they are facing (I have one template per direction).
6. Overlapping duplicate boxes are removed using intersection over union (IoU > 0.5).

*Why not a trained detector like YOLO?* There are only a handful of known sprites, and NCC is unaffected by brightness and contrast. A deep detector would need far more data and would be no more accurate here.

### 4. Working out position: block matching + phase correlation <span class="files">`mapping/registration.py`</span>

- *Main method:* the agent knows which button it pressed, so it can only have moved 0, 1 or 2 cells in that direction. For each option it shifts the old frame by that many 16-pixel cells, measures the mean absolute difference (MAD) with the new frame (ignoring the player's own cell), and picks the smallest.
- *Fallback, phase correlation* (`cv2.phaseCorrelate`), for noisy frames. It takes the Fourier transform of both frames and computes the normalised cross-power spectrum. By the Fourier shift theorem, the inverse transform of that has a sharp peak exactly at the shift.
- *Why both?* Grass and paths repeat every 16 pixels, so phase correlation on its own sometimes mistakes a real step for no movement. It was right 10,766 times out of 10,770. With the pressed direction as a hint, it was right all 10,770 times.

### 5. Planning: region growing + A\* search <span class="files">`planning/explore.py`, `astar.py`</span>

1. Seed a region at the player's cell and keep adding 4-connected neighbours that are walkable. A ledge can only be crossed in the direction you jump off it.
2. The **frontier** is the set of unknown cells touching the grown region.
3. Each frontier cell is scored as (progress north) − 0.5 × (steps to reach it), and the best one becomes the target.
4. A\* search, using a priority queue and the Manhattan distance |dx| + |dy| as its heuristic, finds the path. The first step of the path is the button to press.

### 6. Battles: NCC, glyph matching and pixel counting <span class="files">`perception/battle.py`, `ocr.py`</span>

- *Species:* NCC of the 56×56 enemy sprite box against all 151 Pokémon front sprites. Each sprite is mean-centred and normalised beforehand, so a single matrix multiplication scores all 151 at once.
- *OCR:* the game's text sits on an 8×8 grid in one fixed font. Each 8×8 tile is matched exactly against the font first. If that fails, it takes the closest glyph by **Hamming distance** (the number of pixels that differ), as long as no more than 6 pixels differ. Anything further away isn't text.
- *HP:* count the filled pixels (shade 2 or darker) along the 48-pixel bar and divide by 48.

### 7. Reaching the goal: template matching with a threshold <span class="files">`perception/landmark.py`</span>

- `cv2.matchTemplate` with normalised correlation against the "POKé" sign. A score of 0.9 or more means the agent has arrived.

### 8. Camera mode: localise, restore, rectify, quantise <span class="files">`capture/`</span>

- *Finding the screen* (`localise.py`):
  1. Fit a flat plane (a + bx + cy) to the border of the photo for each colour channel. This is the background model.
  2. Pixels more than 20 grey levels away from that model are screen. This is thresholding.
  3. Clean up the mask (opening, then closing) and take the largest contour.
  4. Find its 4 sides with the Hough line transform.
  5. Refine each side to sub-pixel accuracy by searching the Sobel gradient along the side's normal, then fit a straight line through those points.
  6. The 4 corners are where neighbouring lines intersect.
- *Restoration* (`restore.py`):
  1. Divide by the background plane, which evens out uneven light and colour cast in one step.
  2. Apply Wiener deconvolution with a Gaussian blur kernel. The blur width isn't known, so the code tries several values (σ from 0 to 1.8) and keeps the one whose result looks most like 4 clean shades.
- *Straightening* (`rectify.py`): a 4-point perspective transform maps the corners back to a flat 160×144 screen, sampling at each pixel's centre.
- *Back to 4 shades* (`quantise.py`):
  1. Take the histogram, log-scale it and smooth it.
  2. Find the 4 strongest peaks; the threshold between two shades is the lowest valley between their peaks.

  I also tried multi-Otsu, but it merged rare shades and split spread-out ones, so the valley method won.

### Summary of my choices

| Concept | Common alternatives | What I used | Why |
|---|---|---|---|
| Feature description | SIFT, ORB, HOG | Exact pixel hash | The screen is exact, so no invariance is needed |
| Feature matching | k-nearest neighbours, FLANN | Hash lookup, NCC, Hamming distance | Exact and fast on a fixed palette |
| Object detection | YOLO, Haar cascades | Connected components → masked NCC | Few known sprites, and no training data needed |
| Registration | Feature matching, optical flow | Block MAD + phase correlation | Moves are always whole cells |
| Edge and line detection | Canny, LSD | Sobel + Hough + sub-pixel refinement | Corners to within 0.12 px on level-1 photos |
| Thresholding | Otsu, adaptive | Histogram valleys | Handles rare and spread-out shades |
| Segmentation | Watershed, split-and-merge | Region growing | Grows directly over the tile grid |
| Classification | SVM, k-nearest neighbours | Small CNNs | Learns layout and texture together |
| Restoration | Inverse filter, median | Light flattening + Wiener with blind σ | Best in my measured comparison |
