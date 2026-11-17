# 🏆 Battleship AI Bot: 1st Place, AI Club Battleship Hackathon

**By Jathin Venkumahanthi & Arush Gupta**

An autonomous Battleship bot that won first place in our AI Club's Battleship Bot Hackathon. The bots played each other live over the network in a double-elimination tournament, with a strict **0.5-second limit per shot**.

> Every turn, it simulates thousands of possible enemy fleets and fires where a ship is most likely to be. It hides its own ships where other bots look last, and it learns each opponent's habits as the tournament goes on.

## How it works

### 🎯 Shooting: probabilistic targeting
- Each turn, the bot samples **~6,000 random enemy fleets** that are consistent with every hit, miss and sunk ship so far.
- It counts how often each cell holds a ship across those fleets, which gives a **probability heatmap**, and fires at the hottest cell.
- Hunting and finishing off ships are handled by the same method. After a hit, the neighboring cells light up automatically.
- The sampler is **exactly uniform** over all consistent fleets (vectorized rejection sampling in NumPy). It is unbiased, unlike the starter kit's built-in sampler, and it fits in **< 0.12 s per shot**.

### 🛡️ Defense: adversarial placement
- Probability bots search the center of the board first, because more ship placements fit there.
- Our bot generates 600 random fleets, scores each by how early a probability bot would search its cells, and picks from the **coldest 10%** (edge- and corner-heavy layouts).
- In testing, an unbiased probability bot needed **~47.8 shots** to sink these layouts versus **~44.3** for random ones.
- Every layout mixes horizontal and vertical ships and is **randomly rotated or mirrored**, so it never looks the same twice.

### 🧠 Opponent modeling: learns during the tournament
For each opponent, the bot keeps notes across games:
- **Where they place ships.** A 10×10 weight map leans our shots toward their favorite spots (applied as importance weights on the sampled fleets).
- **Where they search first.** Our ships move away from the cells they fire at early.
- **Repeated layouts.** If an opponent reuses a fleet, the bot fires straight at it and sinks it in the minimum **17 shots**.

### ⚡ Reliability
Every action has a fallback (unbiased sampler → the kit's sampler → exact per-ship counting → a random legal move), so the bot **cannot lose to a crash, a repeated shot or a timeout**.

## Results (local testing)

| Opponent | Result |
|---|---|
| Built-in `hunt-target` bot | ~85–95% win rate |
| Built-in `starter` (random) bot | 100% win rate |
| Bot that reuses the same layout | Sinks it in 17 shots from game 2 on |
| Average shots to sink a full fleet | ~43–46 (vs ~59 for `hunt-target`) |
| Time per shot | ~0.01 s average, ≤ 0.12 s max |

## Running it

The bot plugs into the hackathon kit: [mjessey/battleship-hackathon](https://github.com/mjessey/battleship-hackathon).

1. Download the kit and put `jathin_bot.py` in its root folder (next to `README.md`).
2. Install NumPy: `pip install numpy`
3. Run 10 test games against the built-in hunt-target bot:

**Windows (PowerShell)**
```powershell
$env:PYTHONPATH="src;."; py -3 -m battleship simulate --process --games 10 --bot-a jathin_bot:MyAgent --bot-b hunt-target
```

**macOS / Linux**
```sh
PYTHONPATH=src:. python3 -m battleship simulate --process --games 10 --bot-a jathin_bot:MyAgent --bot-b hunt-target
```

Connect to a tournament server:
```sh
python -m battleship remote-client --server http://HOST:8765 --name jathin --bot jathin_bot:MyAgent
```

## Tunable settings

| Setting | Value | Purpose |
|---|---|---|
| `TARGET_SAMPLES` | 6,000 | Fleets simulated per shot |
| `MAX_SHOT_SECONDS` | 0.12 s | Thinking cap per shot (limit is 0.5 s including network) |
| `PLACEMENT_CANDIDATES` | 600 | Fleets considered when placing ships |
| `PLACEMENT_COLD_FRACTION` | 0.10 | Pick from the coldest 10% of layouts |

Built with Python and NumPy.
