"""stack_tail.py -- full-dims committee snapshot from a recovered prefix + a --tail-from committee.

    python stack_tail.py RECOVERED.pt TAIL_POP.pt K OUT.pt --teacher-from PHASE1_POP.pt
    python stack_tail.py RECOVERED.pt TAIL_POP.pt K OUT.pt [--teacher-seed 0] [--teacher-epochs 25]

RECOVERED.pt: a refine_snapshot_mlp.py checkpoint (dims, act, state_dict) whose layers 1..K are the recovery.
TAIL_POP.pt:  the population snapshot of `run.py --tail-from RECOVERED.pt:K` (recon/_pop__*.pt, dims = dims[K:]).
Each output member = recovered layers 1..K + that tail member's layers. The tail students were trained on exactly
the recovered prefix's output, so the stack is consistent; use the SAME RECOVERED.pt for --tail-from, for this
script, and for refine_snapshot_mlp.py --resume. teacher_state = the real full teacher (cached in teachers/): the
refiner queries it forward only and reads its weights only for the printed error scores. --teacher-from takes it
from a full-dims snapshot (the phase-1 committee's), which guarantees the same black box as the rest of the chain;
otherwise it is loaded from the teachers/ cache next to data.py (and trained if that file is missing).
Run from the code directory (teachers/ and data.py are resolved from there)."""
import argparse, torch
from data import make_teacher

ap = argparse.ArgumentParser()
ap.add_argument("recovered"); ap.add_argument("tail_pop"); ap.add_argument("K", type=int); ap.add_argument("out")
ap.add_argument("--teacher-from", default="", help="full-dims snapshot whose teacher_state to copy (preferred)")
ap.add_argument("--teacher-seed", type=int, default=0); ap.add_argument("--teacher-epochs", type=int, default=25)
a = ap.parse_args()
pre = torch.load(a.recovered, map_location="cpu", weights_only=False); dims = list(pre["dims"]); act = pre["act"]
tp = torch.load(a.tail_pop, map_location="cpu", weights_only=False)
if list(tp["dims"]) != dims[a.K:]:
    raise SystemExit(f"tail dims {tp['dims']} != recovered dims after layer {a.K}: {dims[a.K:]}")
if a.teacher_from:
    t_state = torch.load(a.teacher_from, map_location="cpu", weights_only=False)["teacher_state"]
    if t_state["layers.0.weight"].shape[1] != dims[0] or len(t_state) != 2 * (len(dims) - 1):
        raise SystemExit(f"--teacher-from {a.teacher_from}: teacher does not have dims {dims}")
else:
    t_state = make_teacher(dims, epochs=a.teacher_epochs, seed=a.teacher_seed, device="cpu", verbose=False, act=act).state_dict()
pops = []
for st in tp["pop_states"]:
    sd = {}
    for i in range(a.K):
        sd[f"layers.{i}.weight"] = pre["state_dict"][f"layers.{i}.weight"].float()
        sd[f"layers.{i}.bias"] = pre["state_dict"][f"layers.{i}.bias"].float()
    for i in range(len(dims) - 1 - a.K):
        sd[f"layers.{a.K + i}.weight"] = st[f"layers.{i}.weight"].float()
        sd[f"layers.{a.K + i}.bias"] = st[f"layers.{i}.bias"].float()
    pops.append(sd)
torch.save({"dims": dims, "act": act, "iter": tp.get("iter"),
            "teacher_state": {k: v.detach().cpu() for k, v in t_state.items()}, "pop_states": pops,
            "stacked_from": {"recovered": a.recovered, "tail_pop": a.tail_pop, "K": a.K}}, a.out)
print(f"stacked {len(pops)} members: recovered L1-L{a.K} + tail {tp['dims']} -> {a.out}")
