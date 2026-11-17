"""Jathin's Battleship bot (v3): exact-probability shooting + edge-biased, adaptive defense.

OFFENSE (choose_shot)
  * Every turn: draw thousands of random fleets that are EXACTLY uniform over
    all fleets consistent with every miss, hit and sunk ship (vectorized
    rejection sampling in numpy), then fire at the cell most likely to hold a
    ship. The kit's built-in sampler is biased; this one isn't, and it is ~7x
    faster, so it can afford far more samples.
  * Opponent memory:
      - Every enemy ship we sink is recorded. After a few games, samples are
        importance-weighted toward where THIS opponent likes to put ships
        (e.g. a team that hugs the edges gets its edges searched first).
      - If an opponent ever reuses an exact layout, we replay its stored
        layouts that still fit what we've seen (sinks a fixed fleet in ~17).
  * Fallbacks: the kit's sampler, then exact per-ship counting.

DEFENSE (place_ships)
  * Probability bots assume ships are placed uniformly at random, so they
    search the centre first. Fleets built from the board's "coldest" cells
    (edges/corners) took an unbiased density bot ~47.8 shots vs ~44.3 for a
    random fleet in testing.
  * Adaptive: we record where each opponent fires while hunting and blend
    that into the risk map, so a bot that learns to search edges pushes our
    ships back toward the middle. Randomized among the coldest candidates so
    we are never predictable.

SAFETY
  * Every public method is wrapped: any internal error falls back to a random
    legal action, so the bot can never forfeit from a crash, repeat shot or
    illegal placement. Each shot uses at most 0.12s of the 0.5s limit
    (the rest is headroom for network lag); placement stays under ~1s.

Run:
  battleship simulate --process --bot-a jathin_bot:MyAgent --bot-b hunt-target --games 50
"""

from __future__ import annotations

import random
import time

import numpy as np

from agent import Agent
from core.models import Coord, Outcome, ShipPlacement
from core.rules import Rules, official_rules
from sdk.inference import FleetSamplePool, occupancy_from_fleets
from sdk.observation import ObservationSnapshot
from sdk.placement import random_legal_fleet, ship_cells

SCHEMA = 3
MAX_LAYOUTS = 40

# Sampling / timing knobs.
TARGET_SAMPLES = 6000      # accepted fleets per shot (when time allows)
BATCH = 6000               # proposals per vectorized batch
MIN_SAMPLES = 300          # below this, fall back to the kit sampler
MAX_SHOT_SECONDS = 0.12    # hard cap per shot (limit is 0.5s INCLUDING network)
SHOT_LIMIT_FRACTION = 0.25 # never use more than this share of the per-shot limit
PLACEMENT_MAX_SECONDS = 1.0  # placement limit is 5s; stay far below it

# Defense knobs.
PLACEMENT_CANDIDATES = 600
PLACEMENT_COLD_FRACTION = 0.10


# ---------------------------------------------------------------- geometry
_PLACEMENT_CACHE: dict = {}


def placement_matrix(length: int, rows: int, cols: int) -> np.ndarray:
    """All placements of a ship as a (n, rows*cols) uint8 matrix."""
    key = (length, rows, cols)
    mat = _PLACEMENT_CACHE.get(key)
    if mat is None:
        out = []
        for r in range(rows):
            for c in range(cols - length + 1):
                m = np.zeros(rows * cols, np.uint8)
                m[r * cols + c: r * cols + c + length] = 1
                out.append(m)
        for r in range(rows - length + 1):
            for c in range(cols):
                m = np.zeros(rows * cols, np.uint8)
                m[[(r + i) * cols + c for i in range(length)]] = 1
                out.append(m)
        mat = np.array(out, dtype=np.uint8).reshape(-1, rows * cols)
        _PLACEMENT_CACHE[key] = mat
    return mat


def placement_density(rows: int, cols: int, lengths, blocked: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Per-cell count of single-ship placements avoiding blocked cells.

    Placements covering unresolved hits get a large bonus so target mode
    extends along the line of hits. Used for the empty-board prior and as the
    last-resort fallback.
    """
    heat = np.zeros((rows, cols), dtype=np.float64)
    for length in lengths:
        for horizontal in (True, False):
            r_max = rows if horizontal else rows - length + 1
            c_max = cols - length + 1 if horizontal else cols
            for r in range(r_max):
                for c in range(c_max):
                    sl = (r, slice(c, c + length)) if horizontal else (slice(r, r + length), c)
                    if blocked[sl].any():
                        continue
                    n_hits = int(hits[sl].sum())
                    heat[sl] += 1.0 if n_hits == 0 else 50.0 * n_hits
    return heat


def uniform_heat(lengths, rows, cols, forbidden, hits, rng, deadline, log_weight=None):
    """Exact-uniform posterior occupancy via vectorized rejection sampling.

    forbidden / hits are flat bool arrays. Each alive ship independently picks
    a placement avoiding forbidden cells; a proposal is accepted iff ships do
    not overlap and every unresolved hit is covered. Accepted fleets are
    exactly uniform over all consistent fleets. Optional log_weight (flat,
    per-cell) importance-weights fleets toward an opponent placement prior.
    Returns (heat as flat float array, accepted count).
    """
    mats = []
    for length in lengths:
        mat = placement_matrix(length, rows, cols)
        mat = mat[(mat[:, forbidden] == 0).all(1)] if forbidden.any() else mat
        if len(mat) == 0:
            return None, 0
        mats.append(mat)
    hit_idx = np.flatnonzero(hits)
    heat = np.zeros(rows * cols)
    accepted = 0
    while accepted < TARGET_SAMPLES and time.monotonic() < deadline:
        occ = np.zeros((BATCH, rows * cols), np.uint8)
        for mat in mats:
            occ += mat[rng.integers(0, len(mat), BATCH)]
        ok = occ.max(1) <= 1
        if len(hit_idx):
            ok &= occ[:, hit_idx].all(1)
        good = occ[ok]
        if len(good) == 0:
            continue
        accepted += len(good)
        if log_weight is not None:
            w = np.exp(good @ log_weight)
            heat += w @ good
        else:
            heat += good.sum(0)
    return heat, accepted


# ---------------------------------------------------------------- agent
class MyAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.rules = official_rules()
        self.pool: FleetSamplePool | None = None
        self.memory: dict = {}
        self._sunk_slots: dict[int, set[Coord]] = {}
        self._reset_match_state()

    @classmethod
    def seeded(cls, seed: int | None = None) -> "MyAgent":
        return cls(seed)

    def _reset_match_state(self) -> None:
        self.last_obs: ObservationSnapshot | None = None
        self.last_shot: Coord | None = None
        self.my_hits_open: set[Coord] = set()
        self.opp_hunt_shots: list[Coord] = []
        self.cell_log_weight: np.ndarray | None = None
        self._sunk_slots = {}

    # ------------------------------------------------------------ lifecycle
    def start_match(self, context) -> None:
        try:
            mem = context.memory
            if not isinstance(mem, dict):
                mem = {}
            if mem.get("my_schema") != SCHEMA:
                # Wipe only our own keys; keep the runner's _stats.
                for k in [k for k in mem if not k.startswith("_")]:
                    del mem[k]
                mem["my_schema"] = SCHEMA
            mem.setdefault("games", 0)
            mem.setdefault("opp_ship_counts", [[0] * 10 for _ in range(10)])
            mem.setdefault("opp_ship_cells_total", 0)
            mem.setdefault("opp_ship_games", 0)
            mem.setdefault("opp_layouts", [])  # [{"cells": [[slot, [[r, c], ...]], ...], "seen": n}]
            mem.setdefault("opp_repeats", 0)
            mem.setdefault("opp_hunt_heat", [[0.0] * 10 for _ in range(10)])
            mem.setdefault("opp_hunt_games", 0)
            self.memory = mem
        except Exception:
            self.memory = {}
        self._reset_match_state()

    def observe_opponent_shot(self, result, time_remaining: float) -> None:
        try:
            shot = tuple(result.shot)
            # A "hunt" shot = fired while none of our ships has an open hit;
            # those reveal the opponent's search pattern, not our layout.
            if not self.my_hits_open:
                self.opp_hunt_shots.append(shot)
            if result.outcome is Outcome.HIT:
                self.my_hits_open.add(shot)
            elif result.outcome is Outcome.SUNK and result.sunk_ship is not None:
                for cell in result.sunk_ship.cells:
                    self.my_hits_open.discard(tuple(cell))
        except Exception:
            pass

    def end_match(self, result) -> None:
        try:
            self._learn_from_match(bool(result.won))
        except Exception:
            pass

    # ------------------------------------------------------------ learning
    def _learn_from_match(self, won: bool) -> None:
        mem = self.memory
        if "my_schema" not in mem:
            return
        rules = self.rules
        if (rules.rows, rules.cols) != (10, 10):
            return
        mem["games"] = int(mem.get("games", 0)) + 1

        # Opponent's search pattern: earlier hunt shots = hotter cells.
        if self.opp_hunt_shots:
            heat = mem["opp_hunt_heat"]
            n = len(self.opp_hunt_shots)
            for i, (r, c) in enumerate(self.opp_hunt_shots):
                heat[r][c] += 1.0 - i / max(n, 30)
            mem["opp_hunt_games"] = int(mem.get("opp_hunt_games", 0)) + 1

        # Opponent ship cells we revealed by sinking.
        obs = self.last_obs
        if obs is None:
            return
        sunk_by_slot = {s: set(c) for s, c in self._sunk_slots.items()}
        if won and self.last_shot is not None:
            # The winning shot sank the last ship: its cells are the open hits
            # from the previous turn plus that final shot.
            alive = [s for s in rules.slots if obs.ships_alive[s] and s not in sunk_by_slot]
            if len(alive) == 1:
                final = {tuple(map(int, rc)) for rc in zip(*np.nonzero(obs.hits))}
                final.add(tuple(self.last_shot))
                sunk_by_slot[alive[0]] = final

        counts = mem["opp_ship_counts"]
        for cells in sunk_by_slot.values():
            for r, c in cells:
                counts[r][c] += 1
                mem["opp_ship_cells_total"] = int(mem.get("opp_ship_cells_total", 0)) + 1
        if sunk_by_slot:
            mem["opp_ship_games"] = int(mem.get("opp_ship_games", 0)) + 1

        if len(sunk_by_slot) == len(rules.fleet):
            key = sorted([s, sorted([list(x) for x in cells])] for s, cells in sunk_by_slot.items())
            for entry in mem["opp_layouts"]:
                if entry["cells"] == key:
                    entry["seen"] += 1
                    mem["opp_repeats"] = int(mem.get("opp_repeats", 0)) + 1
                    break
            else:
                mem["opp_layouts"].append({"cells": key, "seen": 1})
                mem["opp_layouts"] = mem["opp_layouts"][-MAX_LAYOUTS:]

    # ------------------------------------------------------------ placement
    def place_ships(self, rules: Rules, time_remaining: float) -> list[ShipPlacement]:
        try:
            return self._place_ships(rules)
        except Exception:
            return random_legal_fleet(rules, self.rng)

    def _place_ships(self, rules: Rules) -> list[ShipPlacement]:
        self.rules = rules
        self.pool = None
        self._sunk_slots = {}
        self._prepare_offense_prior()

        # Risk = how early a probability bot will look at each cell.
        empty = np.zeros((rules.rows, rules.cols), dtype=bool)
        risk = placement_density(rules.rows, rules.cols, rules.fleet, empty, empty)
        risk /= risk.max()

        # Blend in where THIS opponent actually searches first.
        hunt_games = int(self.memory.get("opp_hunt_games", 0)) if self.memory else 0
        if hunt_games >= 2 and (rules.rows, rules.cols) == (10, 10):
            opp = np.array(self.memory["opp_hunt_heat"], dtype=np.float64)
            if opp.max() > 0:
                opp /= opp.max()
                trust = min(0.85, hunt_games / 10.0)
                risk = (1 - trust) * risk + trust * opp

        candidates = []
        place_deadline = time.monotonic() + PLACEMENT_MAX_SECONDS
        for _ in range(PLACEMENT_CANDIDATES):
            if candidates and time.monotonic() > place_deadline:
                break
            fleet = random_legal_fleet(rules, self.rng)
            score = sum(risk[r, c] for p in fleet for (r, c) in ship_cells(rules, p))
            candidates.append((score, fleet))
        candidates.sort(key=lambda x: x[0])
        keep = max(1, int(len(candidates) * PLACEMENT_COLD_FRACTION))
        cold = candidates[:keep]
        # Prefer cold fleets that mix horizontal and vertical ships (2-3 of 5
        # horizontal), so layouts don't all look alike.
        mixed = [c for c in cold if 2 <= sum(p.orientation == "horizontal" for p in c[1]) <= len(c[1]) - 2]
        fleet = self.rng.choice(mixed or cold)[1]
        return self._random_symmetry(rules, fleet)

    def _random_symmetry(self, rules: Rules, fleet: list[ShipPlacement]) -> list[ShipPlacement]:
        """Randomly rotate/mirror the whole layout. The risk map is symmetric,
        so the fleet stays just as hard to find, but looks different each game."""
        try:
            rows, cols = rules.rows, rules.cols
            out = list(fleet)
            if rows == cols and self.rng.random() < 0.5:  # transpose
                out = [ShipPlacement(p.slot, p.col, p.row,
                                     "vertical" if p.orientation == "horizontal" else "horizontal") for p in out]
            if self.rng.random() < 0.5:  # mirror top-bottom
                out = [ShipPlacement(p.slot, rows - 1 - p.row - (rules.fleet[p.slot] - 1 if p.orientation == "vertical" else 0),
                                     p.col, p.orientation) for p in out]
            if self.rng.random() < 0.5:  # mirror left-right
                out = [ShipPlacement(p.slot, p.row,
                                     cols - 1 - p.col - (rules.fleet[p.slot] - 1 if p.orientation == "horizontal" else 0),
                                     p.orientation) for p in out]
            cells = [c for p in out for c in ship_cells(rules, p)]
            if len(cells) == len(set(cells)) and all(0 <= r < rows and 0 <= c < cols for r, c in cells):
                return out
        except Exception:
            pass
        return fleet

    def _prepare_offense_prior(self) -> None:
        """Per-cell log-weights from where this opponent put ships before."""
        self.cell_log_weight = None
        mem = self.memory
        if not mem or (self.rules.rows, self.rules.cols) != (10, 10):
            return
        games = int(mem.get("opp_ship_games", 0))
        if games < 2:
            return
        counts = np.array(mem["opp_ship_counts"], dtype=np.float64).ravel()
        total = counts.sum()
        if total <= 0:
            return
        # Smoothed frequency relative to an average cell (1.0 = average),
        # with 2 pseudo-observations per cell so a few games can't overreact.
        ratio = (counts + 2.0) / (total / counts.size + 2.0)
        beta = min(1.0, games / 8.0)
        self.cell_log_weight = beta * np.log(np.clip(ratio, 0.2, 5.0))

    # ------------------------------------------------------------ shooting
    def choose_shot(self, observation: ObservationSnapshot, time_remaining: float) -> Coord:
        available = observation.available_shots()
        try:
            shot = self._choose_shot(observation, time_remaining)
            if not available[shot]:
                raise ValueError("picked an unavailable cell")
        except Exception:
            cells = [tuple(int(v) for v in rc) for rc in zip(*np.nonzero(available))]
            shot = self.rng.choice(cells)
        self.last_obs = observation
        self.last_shot = shot
        return shot

    def _track_sunk(self, obs: ObservationSnapshot) -> None:
        """Label newly sunk cells with their slot by diffing observations."""
        prev = self.last_obs
        if prev is None:
            return
        newly = [s for s in self.rules.slots if prev.ships_alive[s] and not obs.ships_alive[s]]
        if len(newly) != 1:
            return
        new_cells = obs.sunk.astype(bool) & ~prev.sunk.astype(bool)
        self._sunk_slots[newly[0]] = {tuple(map(int, rc)) for rc in zip(*np.nonzero(new_cells))}

    def _choose_shot(self, obs: ObservationSnapshot, time_remaining: float) -> Coord:
        rules = self.rules
        rows, cols = rules.rows, rules.cols
        self._track_sunk(obs)
        available = obs.available_shots().astype(bool)
        try:
            limit = float(time_remaining)
        except Exception:
            limit = 0.5
        budget = min(MAX_SHOT_SECONDS, max(0.02, limit * SHOT_LIMIT_FRACTION))
        deadline = time.monotonic() + budget

        # 1) Replay a remembered layout if this opponent is known to repeat.
        shot = self._layout_replay_shot(obs, available)
        if shot is not None:
            return shot

        alive = [rules.fleet[s] for s in rules.slots if obs.ships_alive[s]]
        forbidden = (obs.misses | obs.sunk).astype(bool).ravel()
        hits = obs.hits.astype(bool).ravel()

        # 2) Exact-uniform sampling (importance-weighted by opponent prior).
        heat = None
        if budget > 0.005:
            flat, accepted = uniform_heat(
                alive, rows, cols, forbidden, hits, self.nprng, deadline, self.cell_log_weight
            )
            if flat is not None and accepted >= MIN_SAMPLES:
                heat = flat.reshape(rows, cols)

        # 3) Fallback: the kit's constraint-solving sampler (handles rare,
        #    very constrained positions where rejection sampling starves).
        if heat is None and deadline - time.monotonic() > 0.01:
            try:
                if self.pool is None:
                    self.pool = FleetSamplePool(rules, max_samples=300, rng=self.rng)
                self.pool.time_limit = max(0.01, deadline - time.monotonic())
                fleets = self.pool.update(obs)
                if len(fleets) >= 20:
                    heat = occupancy_from_fleets(rules, fleets).astype(np.float64)
            except Exception:
                heat = None

        # 4) Last resort: exact per-ship counting.
        if heat is None or not (heat * available).any():
            heat = placement_density(rows, cols, alive, forbidden.reshape(rows, cols), hits.reshape(rows, cols))

        heat = heat * available
        if not heat.any():
            raise ValueError("empty heatmap")
        best = np.flatnonzero(heat >= heat.max() * (1 - 1e-9))
        idx = int(self.rng.choice(list(best)))
        return (idx // cols, idx % cols)

    def _layout_replay_shot(self, obs: ObservationSnapshot, available: np.ndarray) -> Coord | None:
        mem = self.memory
        if not mem or int(mem.get("opp_repeats", 0)) < 1:
            return None
        misses = obs.misses.astype(bool)
        hit_cells = {tuple(map(int, rc)) for rc in zip(*np.nonzero(obs.hits))}
        best, best_seen = None, 0
        for entry in mem.get("opp_layouts", []):
            layout = {int(s): {tuple(x) for x in cells} for s, cells in entry["cells"]}
            all_cells = set().union(*layout.values())
            if any(misses[r, c] for r, c in all_cells):
                continue
            if not hit_cells <= all_cells:
                continue
            if any(layout.get(slot) != cells for slot, cells in self._sunk_slots.items()):
                continue
            todo = [rc for rc in all_cells if available[rc]]
            if todo and entry["seen"] > best_seen:
                best, best_seen = todo, entry["seen"]
        if best is None:
            return None
        r, c = self.rng.choice(sorted(best))
        return (int(r), int(c))
