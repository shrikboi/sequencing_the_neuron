#!/usr/bin/env bash
# Deep leaky-ReLU 3072x256^8x100 (CIFAR-100 teacher, e25, seed 0): committee + kink refiner, layers 1-7 exact.
# See RESULTS_deep_relu_256x8_2026-11-02.md (what each stage fixes, recorded numbers, what is still open).
#   bash reproduce_relu256x8.sh STAGE [OUT]      STAGE in committee refine15 tail stack refine67 all
# Run on one GPU from this directory (CIFAR-100 under ../data; the teacher is trained/cached in teachers/).
#   committee  phase 1, 8 students, 200k q/it x 30 (6M queries; ~85 min on an L40S). Its replay buffer is 51 GB:
#              set DISK=<dir with ~30 GB free> to stream the old part of the window from disk (needed under ~64 GB RAM).
#   refine15   kink refiner layers 1-5 (design sampler at layers 1, 4, 5; re-anchored guesses).
#   tail       committee on layers 6-9 over the recovered layers 1-5 (real teacher queried at x; 6M queries).
#   stack      recovered layers 1-5 + tail committee -> one full-dims snapshot.
#   refine67   kink refiner layers 6-7 from that snapshot (stops after layer 7: layer 8 still has 4 units no member learns).
# NOTE: this is the clean chain. The recorded layer 1-7 result came from the same steps split over several resumed
# runs (exact path in the RESULTS file); the clean chain has not been run end to end yet. GPU runs are not
# bit-reproducible.
set -euo pipefail
STAGE="${1:?usage: reproduce_relu256x8.sh STAGE [OUT]   STAGE in committee refine15 tail stack refine67 all}"
OUT="${2:-runs/relu256x8}"
cd "$(dirname "$0")"
PY="${PY:-python}"
mkdir -p "$OUT/logs" recon results
ARCH=3072,256,256,256,256,256,256,256,256,100
TAG=r256x8
POP=recon/_pop__mergedbest512_${TAG}__3072x256x256x256x256x256x256x256x256x100__s0.pt
TAILPOP=recon/_pop__mergedbest512_${TAG}tail5__256x256x256x256x100__s0.pt
L5="$OUT/p2_L1-5/refined.pt.L5.pt"
STACKED="$OUT/stacked_L5_tail5.pt"
# refiner flags (all off by default in refine_snapshot_mlp.py):
#   --reanchor       each layer's guess rewritten over the refined previous layer (activation matching, no queries)
#   --dup-check      refined rows that coincide (a rare unit refined onto a copy of another) are re-solved
#   --orphan-gauge net  committee rows for failed units mapped onto the refined prefix, not onto the guess member
#   --design-from 4  layers >= 4 solved with the design sampler (precision ~2x/layer instead of 10-50x for tracking)
#   --design-unc-gate 1e-4  design fits accepted below uncertainty 1e-4 (1e-5 rejects correct rows at depth)
RF="--unit-report --reanchor --dup-check --orphan-gauge net --design-from 4 --design-unc-gate 1e-4"
run() { echo "== $*"; "$@"; }
st_committee() {
  run $PY -u run.py --arch $ARCH --variant mergedbest512 --device cuda --seed 0 --teacher-seed 0 --outer 30 --window 20 \
    --q 200000 --qg-chunk-gb 2 --balance 1 --balance-refresh 5 --cluster-eps 0.1 --cluster-eps-l1 0.02 --masked-consensus \
    --pop-save-every 5 --no-save-recon --tag $TAG ${DISK:+--disk "$DISK" --disk-ram-gb 24 --disk-seg-gb 2} 2>&1 | tee "$OUT/logs/committee.log"
}
st_refine15() {
  mkdir -p "$OUT/p2_L1-5"
  run $PY -u refine_snapshot_mlp.py "$POP" --out "$OUT/p2_L1-5/refined.pt" --stop-after 5 $RF 2>&1 | tee "$OUT/logs/refine15.log"
}
st_tail() {
  run $PY -u run.py --arch 256,256,256,256,100 --tail-from "$L5:5" --variant mergedbest512 --device cuda --seed 0 --teacher-seed 0 \
    --outer 30 --window 20 --q 200000 --balance 1 --balance-refresh 5 --cluster-eps 0.1 --masked-consensus \
    --h-push 30 --h-push-mode dir --pop-save-every 5 --no-save-recon --tag ${TAG}tail5 2>&1 | tee "$OUT/logs/tail.log"
}
st_stack() { run $PY stack_tail.py "$L5" "$TAILPOP" 5 "$STACKED" --teacher-from "$POP" 2>&1 | tee "$OUT/logs/stack.log"; }
st_refine67() {
  mkdir -p "$OUT/p2_L6-7"
  run $PY -u refine_snapshot_mlp.py "$STACKED" --out "$OUT/p2_L6-7/refined.pt" --resume "$L5" --start-layer 6 --stop-after 7 $RF \
    2>&1 | tee "$OUT/logs/refine67.log"
}
case "$STAGE" in
  committee|refine15|tail|stack|refine67) st_$STAGE ;;
  all) st_committee; st_refine15; st_tail; st_stack; st_refine67 ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
