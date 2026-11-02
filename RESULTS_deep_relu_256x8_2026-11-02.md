# Deep leaky-ReLU 3072x256^8x100: challenges and fixes (2026-11-02)

Goal (2026-10-30): on deep ReLU nets, phase 1 (committee disagreement) recovers only ~97% of the neurons;
phase 2 (kink refinement) should be able to finish from there. Target setting: 3072 -> 256^N -> 100.

**Status: layers 1-7 exact, layer 8 252/256, output layer not yet fitted.** One teacher, one seed.

## Setting

- Teacher: leaky ReLU (slope 0.01) MLP `3072,256,256,256,256,256,256,256,256,100`, CIFAR-100, 25 epochs, seed 0
  (`teachers/teacher_3072x256x256x256x256x256x256x256x256x100_e25_s0.pt`, md5 `4b249ab53d1154ac9dd3bbc4491130f1`).
- Black box: forward queries only. Teacher weights are read only for the reported errors (`param_errors`,
  scale/permutation aligned, per matrix) and for the eval-only diagnostics.
- Phase 1: `run.py --variant mergedbest512`, 8 members, 200k queries/iteration x 30, window 20, cluster-eps 0.1
  (0.02 at layer 1), masked consensus, balance 1/5 -- the recipe of the earlier 9-layer (width 200) run.
- Phase 2: `refine_snapshot_mlp.py` (the existing kink refiner). For a hidden unit of layer l, every point x on its kink
  satisfies w.h_{l-1}(x) + b = 0; with layers < l exact, h_{l-1}(x) is computable, so ~2*Din kink points give the
  row as the null vector of [h 1]. Layer by layer, starting from one committee member's rows as guesses
  ("guess member", chosen once by committee agreement), angle gate 12 deg, stops at the first incomplete layer.

## Results

| layer | max eps W / b | queries, cumulative | how the layer was solved |
|---|---|---|---|
| L1 | 1.9e-11 / 3.2e-10 | 21.0M | design sampler (wide-input sweeps), 7 units from committee rows |
| L2 | 3.9e-9 / 7.6e-10 | 36.0M | tracking; duplicate (68,183) found by `--dup-check`, missing unit re-solved from a committee row |
| L3 | 3.4e-8 / 2.6e-9 | 61.2M | tracking + design fallback; 2 units from committee rows (`--orphan-gauge net`) |
| L4 | 1.8e-7 / 1.3e-8 | 151.2M | design sampler (before `--reanchor` existed: 90M for this layer) |
| L5 | 1.3e-6 / 5.2e-8 | 173.8M | design sampler, re-anchored guesses (22.7M) |
| L6 | 4.2e-6 / 3.2e-8 | 196.4M | design sampler, guesses from the tail committee (22.6M) |
| L7 | 7.3e-6 / 3.4e-8 | 219.0M | design sampler, 1 committee row, duplicate (29,62) re-solved (22.5M) |
| L8 | 252/256 rows | +~28M | design sampler with `--design-unc-gate 1e-4`; 4 units nobody learned (see Open) |

Other queries: phase 1 6.0M; tail committee 6.0M; the tracked layer 5 the tail committee was trained on 7.4M.
Total for layers 1-7: ~238M.

Precision with tracking instead (same chain, `--reanchor`, run F): L2 5.8e-9, L3 4.3e-8, L4 1.8e-6, L5 2.8e-5,
L6 2.3e-4 -- too coarse for layer 7 (a correct row's fit uncertainty 3.8e-4 fails the 1e-5 gate).

## Challenges and fixes (in the order they were hit)

### 1. Phase 1 stops at ~99%: rare-firing deep units
Plain committee: 2023/2048 units in consensus after 6M queries (83 min); held by >= 5/8 members at cos > 0.99:
1995. 13 units were learned by no member (L2 teacher 236, L6 6, L7 84, L8 91, 100, 128, 130, 151, 166, 182, 199,
237, 253); almost all fire on <= 0.2% of N(0, 0.25) inputs, so queries rarely cross their kinks. About 40 more are
held by only 1-4 members.

### 2. Silent duplicate rows -> next layer 0/256 (`--dup-check`, commit b782ae8)
For a unit nobody learned, the guess member's row was a rough copy of another unit; tracking refined it onto that
unit exactly (rows 68, 183 of L2: cos 1 - 2e-16). The second-track exactness check agrees with itself, so it passed.
Two identical columns of h make e_68 - e_183 a null direction of every next-layer fit: every L3 row came back as the
true row plus ~1/sqrt(2) of it (89 deg from its guess) and 0/256 passed. Eval-only check: replacing only that unit
by the true row -> 254/256 L3 rows accepted. Fix: after each layer, rows that coincide (cos > 1 - 1e-6, no queries)
are a duplicate; the copy farther from its own guess is re-solved (committee rows, then guess-free) and must come
back distinct (tolerance 2e-3 -- at depth two solves of one unit differ by ~1e-4).

### 3. Committee-row fallback scrambled (`--orphan-gauge net`, b782ae8)
Rows of the other members were aligned to the guess member; with a broken unit in the guess member that alignment
distorts them (best row for the missing unit: 31.5 deg instead of 13.6) and discards good ones as "already solved".
Fix: every member's rows are mapped onto the REFINED previous layer by matching unit activations on random inputs
(exact up to scale for leaky ReLU, no queries).

### 4. Every layer's guess in the wrong coordinates (`--reanchor`, dba5398)
The refiner was handed the guess member's raw rows, which weight that member's own previous-layer units. Wherever a
unit had been replaced (committee rows, duplicate repairs), those columns point at the wrong unit: the L5 guess was
15 deg off (median; 54/256 within the 12 deg gate) while the same member's rows mapped onto the refined L4 were 0.12
deg off (max 1.4). Tracking then lost the kink surface and the design sampler found ~1 kink point per unit per round
(400M+ queries for 44 L5 units). Fix: before each layer, rewrite the guess rows over the refined previous layer by
activation matching; columns for refined units the guess member has no unit for, or two of, are filled per column
from the median of the other members that supply them. Effect: L5 by tracking in 69 s / 7.4M queries; design
sampler ~60 points per unit per round; L8 direct acceptances 54 -> 163/256.

### 5. Units nobody learned: guess-free recovery fails at depth -> tail committee (`--tail-from`, dba5398)
The refiner's guess-free scan (unexplained kinks, normal mapped back through the prefix Jacobian) at L6: 53M queries,
~7900 candidate rows, all rejected (no cluster recurred). Fix: freeze the recovered layers 1-5 and run a fresh
committee on layers 6-9 only (`run.py --tail-from RECOVERED.pt:5 --arch 256,256,256,256,100`, `tail_bb.py`).
Queries are mined in raw x through the recovered prefix; the black box is the real teacher at that x (h -> x hash
lookup; an h it did not produce raises). 6.0M queries, 36 min: every L6 and L7 unit held (L7 84 by one member at
cos 0.9998), L8 242/256 held -- 6 of the 10 L8 units nobody learned now held; still missing 128, 130, 166, 199.
`stack_tail.py` stacks the tail committee onto the recovered prefix for the refiner.

### 6. Precision lost per layer (`--design-from`, `--design-unc-gate`, dba5398)
Tracking loses 10-50x per layer (above); the design sampler ~2x and, with re-anchored guesses, costs ~23M queries
per 256-unit layer. `--design-from K` uses it for every row from layer K. Its fixed uncertainty gate 1e-5 then
rejected correct L8 rows on the 7e-6 prefix (uncertainty median 1.5e-5, angle 0.2 deg, validation 1.00; a wrong
lock is ~1e-2): `--design-unc-gate 1e-4` -> L8 253/256 direct.

### 7. Smaller
- `orphans()` crashed on a layer with no solved row (empty `amax`); guarded (dba5398).
- `peel/informative_kinks.recover_layer` takes `unc_gate` / `gap_gate` kwargs (defaults unchanged).

## What did not work / did not help

- **Guess-free scan** (Jacobian mapback): 0 usable rows at L2 (48.7M q) and L6 (53M q).
- **`--prefix-check`** (probe the next layer for a coordinate many fits load on): detects the duplicate correctly,
  but its guess-free repair did not recover the unit. Superseded by `--dup-check`.
- **Second tail committee on layers 8-9 only** (`--tail-from L7:7 --arch 256,256,100`): worse than the 6-9 one --
  206/256 L8 units held, 29 held by nobody (incl. 128, 130, 166, 199), loss stuck at 0.37, many sign-flipped
  members (leaky(-z) = leaky(z) - 1.01 z: a flipped unit differs only by a linear term).

## Open

1. **L8 units 128, 130, 166, 199** (no committee member learns them). Proposed: L8 is the last hidden layer and the
   head is linear, so with L1-L7 exact and 252 L8 rows exact every unexplained kink of the black box belongs to one
   of the 4; each kink point x* gives one exact equation w.h7(x*) + b = 0 (no Jacobian). ~300 points per unit,
   split into 4 planes by a consensus fit. Not implemented yet.
2. Output layer: closed-form least squares once L8 is complete (in the refiner).
3. Single guess member: its defects (duplicates) become the refiner's; a per-layer committee guess (median of all
   members' mapped rows: 0.20 deg at L5 vs 0.12 for the guess member alone) is untested.
4. Only one teacher / seed; the clean chain below has not been run end to end.

## How to run

`reproduce_relu256x8.sh STAGE [OUT]` (stages committee, refine15, tail, stack, refine67, all; see its header).
Refiner flags: `--reanchor --dup-check --orphan-gauge net --design-from 4 --design-unc-gate 1e-4` (all off by
default). The phase-1 replay buffer is 51 GB: set `DISK=<dir>` for the disk ring when RAM < ~64 GB.

The recorded numbers above came from these steps split over resumed runs
(`/home/shirmao/data/games/runs/relu256x8/`, logs in `logs/`):
1. `p1_plain`: phase 1 (`run.py ... --tag r256x8 --disk $TMPDIR/ring --disk-ram-gb 24 --disk-seg-gb 2`).
2. `p2_plain`: refiner L1 (unchanged driver).
3. `p2_plain_dup3` / `p2_plain_dup4`: L2-L3, `--dup-check --orphan-gauge net`.
4. `p2_plain_dA`: L4, `--design-from 4` (no re-anchoring yet).
5. `p2_plain_E`: L5 by tracking with `--reanchor` (the prefix the tail committee was trained on).
6. `tail_run`: `run.py --tail-from p2_plain_E/refined.pt.L5.pt:5 --arch 256,256,256,256,100 ... --h-push 30
   --h-push-mode dir --tag r256x8tail5`.
7. Stacked with the run folder's copy of `stack_tail.py`; the committed
   `stack_tail.py p2_plain_E/refined.pt.L5.pt <tail pop> 5 <stacked> --teacher-from <phase-1 pop>` reproduces that
   snapshot bit for bit (members and teacher).
8. `p2_tail5d`: refiner on the stacked snapshot, `--resume p2_plain_dA/refined.pt.L4.pt --start-layer 5
   --design-from 5 --reanchor --dup-check --orphan-gauge net` -> L5-L7.
9. `p2_t5d8g`: L8 from `p2_tail5d/refined.pt.L7.pt`, `--design-unc-gate 1e-4` -> 252/256 (stopped by hand).

Eval-only diagnostic scripts (not needed to run the method) are in `runs/relu256x8/tools/`.
