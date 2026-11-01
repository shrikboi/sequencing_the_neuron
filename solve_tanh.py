#!/usr/bin/env python
"""solve_tanh.py -- ONE command: committee -> layer-1 pipeline -> K-FAC refinement stages -> final
unfrozen polish, printing max / mean parameter error and the query count after every stage.

  python solve_tanh.py --arch 3072,200,200,200,200,200,200,200,200,200,100 --tag eleven_a --gpus 0,2 \
      --q 1800000 --window 20 --epochs 5 --disk rings/eleven_a --qg-block-gb 1.61 --qg-chunk-gb 6.5

Stages (each skippable / resumable):
  1 committee     run.py (mergedbest512, tanh, --layer-consensus --align-inverse) on gpus[0]
  2 l1 pipeline   peel/diag/l1_pipeline.py: isolation (sharded over gpus) + gap-fill + orphan recovery +
                  tail-decay scales + assemble + K-FAC refine (layer 1 frozen)   [--refine-cg/--refine-iters]
  3 polish        peel/polish_full.py, K-FAC, layer 1 frozen, bigger CG budget   [--polish-cg/--polish-iters]
  4 final         peel/polish_full.py, K-FAC, everything free                     [--final-cg/--final-iters]
--subspace         run the committee on the measured row space of W1 (run.py --subspace).
--subspace-refine  draw the refiner's probe directions inside that row space in every refinement stage (2b, 3, 4),
                   then add stage 4b: the same polish with full-space directions and only layer 1 free (layers 2.. frozen).
                   U comes from a --subspace committee (no extra queries) or is measured here (counted).
Shallow nets (committee already has layer 1: its W1 max eps < --l1-threshold) skip stages 2-3.
--skip-committee reuses an existing run's files (same --tag). Summary table at the end."""
import argparse, json, os, re, subprocess, sys, time

P = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--gpus", default="0", help="comma list; gpus[0] runs the committee and the solvers, all run isolation shards")
ap.add_argument("--variant", default="mergedbest512"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--teacher-seed", type=int, default=0)
ap.add_argument("--q", type=int, default=300000); ap.add_argument("--window", type=int, default=30); ap.add_argument("--outer", type=int, default=30)
ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--disk", default=None); ap.add_argument("--disk-ram-gb", type=float, default=None)
ap.add_argument("--qg-block-gb", type=float, default=None); ap.add_argument("--qg-chunk-gb", type=float, default=None); ap.add_argument("--qg-tf32", action="store_true")
ap.add_argument("--extra", default="", help="extra run.py flags, quoted")
ap.add_argument("--skip-committee", action="store_true"); ap.add_argument("--skip-l1", action="store_true", help="force the shallow path (no isolation/scales)")
ap.add_argument("--l1-threshold", type=float, default=1e-2, help="--l1-gate oracle only: committee W1 TRUE max eps below this -> shallow path")
ap.add_argument("--l1-gate", choices=["consensus", "oracle"], default="consensus",
                help="consensus (default, black-box): shallow path iff >= --l1-cons-frac of layer-1 units reached committee quorum "
                     "consensus (students only), and the layer-1 pipeline gates units by black-box error. oracle: the original "
                     "truth-based decisions (W1 error vs the teacher), for comparison only")
ap.add_argument("--l1-cons-frac", type=float, default=0.95)
ap.add_argument("--refine-cg", type=int, default=2000); ap.add_argument("--refine-iters", type=int, default=12)
ap.add_argument("--polish-cg", type=int, default=4000); ap.add_argument("--polish-iters", type=int, default=12)
ap.add_argument("--final-cg", type=int, default=4000); ap.add_argument("--final-iters", type=int, default=6)
ap.add_argument("--points", type=int, default=512); ap.add_argument("--dirs", type=int, default=64)
ap.add_argument("--keep-ring", action="store_true", help="keep the --disk replay ring after the committee (default: delete it, it is not used by the later stages)")
ap.add_argument("--trunk", choices=["consensus", "best", "consensus-nol1"], default="consensus",
                help="solver starting net: aligned quorum-consensus net (default), the best member, or consensus-nol1 = consensus "
                     "net WITHOUT the layer-1 order check on the deep path (layer 1 is rebuilt there); the shallow path then uses the best member")
ap.add_argument("--subspace", action="store_true", help="committee on the measured first-layer subspace (passes --subspace to run.py)")
ap.add_argument("--subspace-centers", type=int, default=16,
                help="Jacobian centres for measuring the subspace (run.py's committee measurement, or --subspace-refine's if the committee "
                     "did not use --subspace); 16 needed at depth (9 hidden: 3 left W1 2e-3 outside the measured span)")
ap.add_argument("--subspace-fp32", action="store_true", help="measure the subspace with float32 outputs (passes --subspace-fp32)")
ap.add_argument("--subspace-refine", action="store_true",
                help="refinement stages draw their probe directions inside span(U), then a final full-space stage with only layer 1 free")
args = ap.parse_args()
dims = [int(x) for x in args.arch.split(",")]; arch = "x".join(map(str, dims))
gpus = args.gpus.split(","); g0 = gpus[0]
ctag = "_".join([args.tag] + (["sub" + ("fp32" if args.subspace_fp32 else "")] if args.subspace else []))   # run.py appends _sub
name = f"{args.variant}_tanh_{ctag}__{arch}__s{args.seed}"
R, L = "recon", "results/logs"; os.makedirs(L, exist_ok=True)
pop_path, final_path = f"{R}/_pop__{name}.pt", f"{R}/{name}_final.pt"
stages = []; q_total = 0; t_start = time.time()


def say(s): print(f"[solve_tanh] {s}", flush=True)


def sh(cmd, log=None):
    say(f"$ {cmd}" + (f"  > {log}" if log else ""))
    rc = subprocess.run(cmd + (f" > {log} 2>&1" if log else ""), shell=True).returncode
    if rc != 0:
        say(f"stage failed (exit {rc}); see {log}"); sys.exit(rc)


def final_line(log):
    """Parse 'FINAL vs black box: max_eps X | max per matrix [...] | mean per matrix [...]' -> (max, mean)."""
    txt = open(log).read()
    m = re.findall(r"FINAL vs black box: max_eps ([\d.e+-]+) \| max per matrix \[(.*?)\] \| mean per matrix \[(.*?)\]", txt)
    if not m: return None, None
    mx, _, means = m[-1]; means = [float(x.strip(" '")) for x in means.split(",")]
    return float(mx), sum(means) / len(means)


def queries_in(log, pat):
    return sum(int(x) for x in re.findall(pat, open(log).read())) if os.path.exists(log) else 0


def record(stage, mx, mean, q_stage, note=""):
    global q_total
    q_total += q_stage
    stages.append((stage, mx, mean, q_stage, q_total, time.time() - t_start))
    say(f"== {stage}: max eps {mx:.3e} | mean eps {mean:.3e} | queries this stage {q_stage/1e6:.2f}M | cumulative {q_total/1e6:.2f}M | {(time.time()-t_start)/3600:.2f} h{(' | ' + note) if note else ''}")


# ---------------------------------------------------------------- 1. committee
if not args.skip_committee:
    cmd = (f"CUDA_VISIBLE_DEVICES={g0} {P} run.py --variant {args.variant} --tanh --device cuda --arch {args.arch} "
           f"--q {args.q} --window {args.window} --outer {args.outer} --teacher-seed {args.teacher_seed} --seed {args.seed} "
           f"--layer-consensus --align-inverse --log-every 1 --pop-save-every 5 --tag {args.tag}")
    if args.epochs: cmd += f" --epochs {args.epochs}"
    if args.batch: cmd += f" --batch {args.batch}"
    if args.disk: cmd += f" --disk {args.disk}"
    if args.disk_ram_gb: cmd += f" --disk-ram-gb {args.disk_ram_gb}"
    if args.qg_block_gb: cmd += f" --qg-block-gb {args.qg_block_gb}"
    if args.qg_chunk_gb: cmd += f" --qg-chunk-gb {args.qg_chunk_gb}"
    if args.qg_tf32: cmd += " --qg-tf32"
    if args.subspace:
        cmd += f" --subspace --subspace-centers {args.subspace_centers}" + (" --subspace-fp32 --subspace-eps 3e-2" if args.subspace_fp32 else "")
    cmd += f" {args.extra}"
    sh(cmd, f"{L}/committee__{args.tag}.log")
res = json.load(open(f"results/{name}.json"))
if args.disk and not args.keep_ring and os.path.isdir(args.disk) and os.path.abspath(args.disk).startswith(os.path.abspath("rings")):
    import shutil; shutil.rmtree(args.disk, ignore_errors=True)     # the replay ring is not used after the committee
    say(f"freed replay ring {args.disk} (committee done; --keep-ring to keep it)")
w1_max = max(res["final_max_eps_per_matrix"][:2])
record("1 committee", res["final_max_eps"], res["final_mean_eps"], int(res["queries"]),
       f"certified {res['log'][-1].get('n_consensus')}/{res['log'][-1].get('n_total')} | W1 max {w1_max:.2e} | per-matrix max {' '.join(f'{x:.1e}' for x in res['final_max_eps_per_matrix'])}")

# ---------------------------------------------------------------- 1c. subspace for the refinement stages
SUBENV = ""
if args.subspace_refine:
    import torch
    _uf = f"{R}/_subspaceU__{args.tag}.pt"
    _fin = torch.load(final_path, map_location="cpu", weights_only=False)
    if "subspace_U" in _fin:
        torch.save({"U": _fin["subspace_U"].double()}, _uf)
        say(f"--subspace-refine: U ({tuple(_fin['subspace_U'].shape)}) from the --subspace committee, no extra queries -> {_uf}")
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import subspace as _ss
        from nets import MLP
        _pk = torch.load(pop_path, map_location="cpu", weights_only=False)
        _T = MLP(dims, act="tanh"); _T.load_state_dict(_pk["teacher_state"])      # the black box, queried below (counted)
        _dev = "cuda" if torch.cuda.is_available() else "cpu"; _T = _T.to(_dev)
        _c = torch.cat([torch.zeros(1, dims[0]), torch.randn(args.subspace_centers - 1, dims[0], generator=torch.Generator().manual_seed(12345))]).to(_dev)
        _U, _s, _q = _ss.measure(_T, dims[1], _c, 3e-2 if args.subspace_fp32 else 1e-4, torch.float32 if args.subspace_fp32 else torch.float64)
        torch.save({"U": _U.cpu()}, _uf)
        record("1c subspace measurement for the refinement", res["final_max_eps"], res["final_mean_eps"], int(_q),
               f"{args.subspace_centers} centres, gap s{dims[1]}/s{dims[1] + 1} {(_s[dims[1] - 1] / _s[dims[1]]).item() if len(_s) > dims[1] else float('inf'):.3g}")
    SUBENV = f"SUBSPACE_U={_uf} "

# ---------------------------------------------------------------- 2. layer-1 pipeline + frozen refine
_cl = (res["log"][-1].get("consensus_eps_per_layer") or [{}])[0] if res.get("log") else {}
l1_cons, l1_tot = _cl.get("n_cons", 0), _cl.get("n_tot", dims[1])
if args.l1_gate == "consensus":
    shallow = args.skip_l1 or l1_cons >= args.l1_cons_frac * l1_tot
    say(f"layer-1 gate (black-box): {l1_cons}/{l1_tot} units in committee consensus -> {'shallow' if shallow else 'deep'} path "
        f"(scoring only, not used: true W1 max eps {w1_max:.1e})")
else:
    shallow = args.skip_l1 or w1_max < args.l1_threshold
    say(f"layer-1 gate ORACLE (truth-based, comparison only): W1 max eps {w1_max:.1e} -> {'shallow' if shallow else 'deep'} path")
import torch
if shallow:
    say("shallow path, skipping isolation/scales")
    guess = f"{R}/_guess__{args.tag}.pt"
    pop = torch.load(pop_path, map_location="cpu", weights_only=False); fin = torch.load(final_path, map_location="cpu", weights_only=False)
    _cs = None
    if args.trunk == "consensus":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "peel", "diag"))
        from consensus_guess import consensus_guess
        _cs, _ = consensus_guess(pop, fin["state_dict"])
    say(f"solver starting net = {'aligned quorum CONSENSUS net' if _cs is not None else 'best member'}")
    if args.trunk == "consensus":                      # consensus_guess's black-box sanity gate (4096 queries, upper bound)
        record("1b trunk consensus gate", res["final_max_eps"], res["final_mean_eps"], 4096)
    torch.save({"dims": dims, "act": "tanh", "teacher_state": pop["teacher_state"], "state_dict": _cs if _cs is not None else fin["state_dict"]}, guess)
    base = guess
else:
    sh(f"{SUBENV}TRUNK={args.trunk} L1_GATE={'blackbox' if args.l1_gate == 'consensus' else 'oracle'} {P} peel/diag/l1_pipeline.py {args.tag} {pop_path} {final_path} {args.gpus} {args.refine_cg} {args.refine_iters}", f"{L}/l1_pipeline__{args.tag}.log")
    guess = f"{R}/_guess__{args.tag}.pt"
    # every stage prints its oracle's total ("total queries N"): isolation shards + gap-fills + orphan re-isolations,
    # scale shards + orphan scales (incl. their row-space jets), orphan searches, and the trunk gate
    q_iso = sum(queries_in(f"{L}/{f}", r"total queries (\d+)") for f in os.listdir(L) if f.startswith(f"iso_l1_{args.tag}_"))
    q_sc = sum(queries_in(f"{L}/{f}", r"total queries (\d+)") for f in os.listdir(L) if f.startswith(f"scale_l1_{args.tag}_"))
    q_orph = sum(queries_in(f"{L}/{f}", r"orphan search total queries (\d+)") for f in os.listdir(L) if f.startswith(f"orphan_l1_{args.tag}_"))
    q_gate = queries_in(f"{L}/l1_pipeline__{args.tag}.log", r"trunk gate total queries (\d+)")
    asm = re.findall(r"assembled net vs black box: max_eps ([\d.e+-]+) .*?mean per matrix \[(.*?)\]", open(f"{L}/refine_l1fixed__{args.tag}.log").read())
    if asm:
        means = [float(x.strip(" '")) for x in asm[-1][1].split(",")]
        record("2a layer-1 isolation + scales (assembled, before refine)", float(asm[-1][0]), sum(means) / len(means), q_iso + q_sc + q_orph + q_gate,
               f"isolation {q_iso/1e6:.2f}M scales {q_sc/1e6:.2f}M orphans {q_orph/1e6:.2f}M gate {q_gate}")
    mx, mean = final_line(f"{L}/refine_l1fixed__{args.tag}.log")
    record(f"2b K-FAC refine, L1 frozen{', subspace dirs' if SUBENV else ''} (cg {args.refine_cg} x {args.refine_iters})", mx, mean, queries_in(f"{L}/refine_l1fixed__{args.tag}.log", r"-> (\d+) queries"))
    base = f"{R}/_l1fixed_refined__{args.tag}.pt"
    # 3. frozen polish with the bigger budget
    if args.polish_iters > 0:
        out = f"{R}/_polished__{args.tag}.pt"
        sh(f"{SUBENV}FREEZE=layers.0.weight PRECOND=kfac CUDA_VISIBLE_DEVICES={g0} {P} peel/polish_full.py {guess} {base} {out} {args.points} {args.dirs} {args.polish_iters} {args.polish_cg}", f"{L}/polish__{args.tag}.log")
        mx, mean = final_line(f"{L}/polish__{args.tag}.log")
        record(f"3 K-FAC polish, L1 frozen{', subspace dirs' if SUBENV else ''} (cg {args.polish_cg} x {args.polish_iters})", mx, mean, queries_in(f"{L}/polish__{args.tag}.log", r"-> (\d+) queries"))
        base = out

# ---------------------------------------------------------------- 4. final polish, everything free
if args.final_iters > 0:
    out = f"{R}/_final{'_sub' if SUBENV else ''}__{args.tag}.pt"
    lg = f"{L}/final{'_sub' if SUBENV else ''}__{args.tag}.log"
    sh(f"{SUBENV}PRECOND=kfac CUDA_VISIBLE_DEVICES={g0} {P} peel/polish_full.py {guess} {base} {out} {args.points} {args.dirs} {args.final_iters} {args.final_cg}", lg)
    mx, mean = final_line(lg)
    record(f"4 K-FAC final polish, all free{', subspace dirs' if SUBENV else ''} (cg {args.final_cg} x {args.final_iters})", mx, mean, queries_in(lg, r"-> (\d+) queries"))
    base = out
    if SUBENV:
        # 4b: directions inside span(U) leave the first layer's biases (and its weights' out-of-span part) poorly determined
        # (5 hidden: b1 stuck at 5.6e-5 while layers 2-6 reached 1e-7). Same polish, full-space directions, only layer 1 free.
        out = f"{R}/_final__{args.tag}.pt"
        frz = ",".join(f"layers.{i}.{p}" for i in range(1, len(dims) - 1) for p in ("weight", "bias"))
        sh(f"FREEZE={frz} PRECOND=kfac CUDA_VISIBLE_DEVICES={g0} {P} peel/polish_full.py {guess} {base} {out} {args.points} {args.dirs} {args.final_iters} {args.final_cg}", f"{L}/final__{args.tag}.log")
        mx, mean = final_line(f"{L}/final__{args.tag}.log")
        record(f"4b K-FAC polish, layer 1 only, full-space dirs (cg {args.final_cg} x {args.final_iters})", mx, mean, queries_in(f"{L}/final__{args.tag}.log", r"-> (\d+) queries"))
        base = out

print("\n" + "=" * 100)
print(f"SUMMARY {name}  ({(time.time()-t_start)/3600:.2f} h)")
print(f"{'stage':62s} {'max eps':>10s} {'mean eps':>10s} {'queries':>10s} {'cumul':>10s} {'hours':>6s}")
for st, mx, mean, qs, qc, t in stages:
    print(f"{st:62s} {mx:10.3e} {mean:10.3e} {qs/1e6:9.2f}M {qc/1e6:9.2f}M {t/3600:6.2f}")
print(f"final net: {base}")
print("=" * 100, flush=True)
