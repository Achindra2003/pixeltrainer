# PixelTrainer: A Vision-Only Agent that Plays Pokémon from Screen Pixels

## a. Synopsis

Game-playing AI usually "cheats": it reads the game's internal memory, or uses a large language model that is slow and needs helper tools. Recent "LLM plays Pokémon" experiments showed that general AI models still struggle to *see* a game screen reliably. **PixelTrainer** takes the opposite route. It plays Pokémon (Game Boy, run in the PyBoy emulator) using **only the screen pixels**, through a purpose-built computer-vision pipeline, much like a human player.

The task is to travel from one town to the next (for example, Pallet Town to Viridian City) and win the wild and trainer battles along the way. The pipeline:

1. **Screen-state classification.** A small CNN (a basic image-classification network) decides whether the frame shows the overworld, a battle, a dialogue box or a menu.
2. **Overworld perception.** Each 16×16 tile is classified as walkable, wall, tall grass, water, door, ledge or NPC. NPCs and their facing direction are detected, since trainers spot you by line of sight.
3. **Map building.** The agent estimates how far the view shifts between frames by image registration (aligning each frame with the previous one), and stitches the screens into a growing global map, a simplified version of the SLAM (Simultaneous Localisation and Mapping) that robots use. It also detects map changes when the player enters a building.
4. **Planning.** A* path-finding over the map built so far, with costs for tall grass and one-way ledges, and exploration when the goal is not yet visible.
5. **Battle perception and decisions.** The opponent's Pokémon is recognised from its sprite, HP bars are read by colour segmentation, and a custom OCR tuned to the game's fixed font reads the moves and battle text. Moves are chosen using the type chart and an expected-damage estimate.

The novelty is making **explicit visual perception, map building and planning** the whole agent, with no memory reads while it plays, and **measuring how accurately it perceives the screen** against the game's true internal state.

## b. Input to the System

| Input | Why it is needed |
|---|---|
| **Live emulator frames** (160×144 pixels, about 60 frames per second) | These are the only thing the agent observes, exactly as a human sees the game. Every decision has to come from pixels. |
| **Sprite and tile reference images** taken from the game | Recognising Pokémon species and tile types needs one labelled example per class, so no large training dataset is required. |
| **Game RAM values** (player position, map ID, HP, species), **used for evaluation only** | Community-documented memory addresses give the true game state. This lets the perception be scored objectively, and the agent never uses it for decisions. |
| **Goal specification** (a target town or map) | It defines the navigation task and when the task counts as complete. |

## c. Expected Output

| Output | Why this result is expected |
|---|---|
| **An agent playing live**, with button presses generated from pixels | The game screen is fully visible and has a regular grid structure, so a pipeline of classification, registration and planning can extract enough state to act on. |
| **A live view of what the agent sees**: tile labels, the growing map, the planned path, and battle reasoning (e.g. "Thunderbolt: 2× effective") | This makes every decision explainable and debuggable, unlike black-box AI players. |
| **A perception accuracy report** comparing the agent's readings with the true game state: tile-classification accuracy, map-position error in tiles, species-recognition accuracy, HP-reading error | The true state is available from RAM, so accuracy can be measured exactly rather than judged by eye. |
| **Task results**: success rate reaching the goal, steps taken against the shortest path, battle win rate against random and greedy baselines, and decisions per second | These show that better perception actually leads to better play, and the baselines put the numbers in context. |

## d. Conclusion

PixelTrainer turns a familiar game into a controlled test of a full computer-vision system: classification, template and embedding-based recognition, OCR, colour segmentation, image registration, map building and planning, all evaluated against exact ground truth. It runs on an ordinary laptop with free, open-source tools, needs no large training dataset, and can be demonstrated live, with the game on one side of the screen and the agent's view of the world on the other. Future scope includes the Game Boy Advance games (FireRed/Emerald, with richer colour graphics), long-horizon goals such as winning gym badges, and a reinforcement-learning policy built on the same perception system.

*Note: the game ROM is supplied by the user from a legally owned cartridge and is not distributed with the project.*
