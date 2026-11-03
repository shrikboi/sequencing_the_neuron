#!/usr/bin/env bash
# Reproduce the deep tanh MLP recoveries (results/RECORD_tanh_demo_campaign_2026-10-16.md), one depth per call.
#   bash reproduce_tanh_mlp.sh DEPTH [GPU] [TAG]          DEPTH in 5L 6L 8L 9L 10L 11L 12L 13L 14L 15L
# Teacher seed 0, input 3072, hidden width 200, output 100 ("8L" = 8 layers incl. input and output = 6 hidden).
# Pipeline (solve_tanh.py): committee -> layer-1 isolation + tail-decay scales (deep nets only) -> K-FAC jet refine
# (layer 1 frozen) -> K-FAC polish (frozen) -> K-FAC final polish (all free).  5L/6L also get the extra 8-step
# all-free polish of the record.  Recorded results (max / mean eps, total queries, wall on a shared box):
#   5L  4.2e-11 / 2.8e-12   1.75M    ~1.5 h        8L   1.9e-6 / 8.9e-9   137.4M   ~7.7 h
#   6L  3.9e-8  / 1.4e-9    3.24M    ~1.5 h        9L   1.7e-6 / 3.4e-8   182.9M   ~14 h
#                                                  10L  2.9e-6 / 6.2e-8   181.5M   ~14 h
# 8L-10L use the --disk replay ring: free disk needed ~130 GB (8L), ~240 GB (9L), ~330 GB (10L) under rings/.
#
# 11L-15L: the --subspace pipeline. The committee runs on the measured row space of W1 (16 Jacobian centres,
# 98,304 queries; 15L: 32 centres, 196,608 queries; run.py --subspace), window 20, 40 rounds, 5 epochs; the refinement stages draw their directions in
# that subspace (--subspace-refine), with a final layer-1-only stage in the full input space. Black-box layer-1 gate
# (--l1-gate consensus); starting net = the best member (--trunk consensus-nol1: on these depths the committee solves
# layer 1, so the shallow path runs, without isolation/scales). Then an extra polish of the solved net:
#   A: all free, subspace directions, 8 K-FAC steps (cg 4000)      B: layer 1 only, full-space directions, 4 steps (cg 4000)
# (+131,584 queries each). Results (L40S, seed 0; committee | after the extra polish, max / mean eps, total queries):
#   11L  q  450k/round   committee 1.86e-4 / 2.17e-5   final 5.81e-5 / 1.47e-6   18.62M   (~1.7 h + 1.2 h + 1.3 h)
#   12L  q  600k/round   committee 2.50e-4 / 2.43e-5   final 7.96e-5 / 4.59e-6   24.62M   (~2.3 h + 1.3 h + 1.4 h)
#   13L  q  800k/round   committee 2.51e-4 / 2.70e-5   final 9.26e-5 / 8.65e-6   32.62M   (~3.3 h + 1.3 h + 1.5 h)
#   14L  q 1000k/round   committee 3.50e-4 / 3.27e-5   final 2.14e-4 / 1.66e-5   40.62M   (~4.3 h + 1.4 h + 2.0 h)
#   15L  q 2400k/round   committee 3.43e-4 / 3.69e-5   final 3.16e-4 / 2.23e-5   96.72M   (~8.5 h + 1.5 h + 4.8 h)
# Queries = committee (rounds x q + 98,304; 15L + 196,608) + 4 x 131,584 (solve: subspace polish + layer-1 polish;
# extra polish A + B). 11L-14L keep the committee's query window in RAM (~6-24 GB host memory); at the end run.py
# writes the window to recon/mergedbest512_tanh_<TAG>_sub__200x...pt (11-29 GB), unused later, so this script deletes
# it. 15L's window (58 GB) does not fit: it streams from the --disk ring (8 GB RAM tier, ~60 GB free disk under
# rings/, deleted after the committee); the RAM-tier size sets which rows rank the members, so keep it at 8.
# GPU runs are not bit-reproducible: expect numbers close to, not equal to, the record.
set -euo pipefail
DEPTH="${1:?usage: reproduce_tanh_mlp.sh DEPTH [GPU] [TAG]}"; GPU="${2:-0}"; TAG="${3:-repro_$DEPTH}"
cd "$(dirname "$0")"
mkdir -p results/logs recon rings
S="--refine-cg 2000 --refine-iters 8 --polish-cg 2000 --polish-iters 6 --final-cg 2000 --final-iters 4"
DISK="--disk rings/$TAG --disk-ram-gb 50"
BIG="--window 20 --epochs 5 $DISK --qg-block-gb 1.61 --qg-chunk-gb 6.5"
SUB="--window 20 --outer 40 --epochs 5 --subspace --subspace-centers 16 --subspace-refine --l1-gate consensus --trunk consensus-nol1"
SUBDEPTH=0
case "$DEPTH" in
  5L)  ARCH=3072,200,200,200,100;                         OPTS="--q 48000 --window 30" ;;
  6L)  ARCH=3072,200,200,200,200,100;                     OPTS="--q 96000 --window 30" ;;
  8L)  ARCH=3072,200,200,200,200,200,200,100;             OPTS="--q 300000 --window 30 $DISK" ;;
  9L)  ARCH=3072,200,200,200,200,200,200,200,100;         OPTS="--q 1200000 $BIG" ;;
  10L) ARCH=3072,200,200,200,200,200,200,200,200,100;     OPTS="--q 1200000 $BIG" ;;
  11L) Q=450000 ;;
  12L) Q=600000 ;;
  13L) Q=800000 ;;
  14L) Q=1000000 ;;
  15L) Q=2400000; SUB="${SUB/--subspace-centers 16/--subspace-centers 32} --disk rings/$TAG --disk-ram-gb 8" ;;
  *) echo "unknown depth $DEPTH (5L 6L 8L 9L 10L 11L 12L 13L 14L 15L)"; exit 1 ;;
esac
case "$DEPTH" in
  11L|12L|13L|14L|15L)
    SUBDEPTH=1; NH=$(( ${DEPTH%L} - 2 )); ARCH="3072$(printf ',200%.0s' $(seq $NH)),100"; OPTS="--q $Q $SUB" ;;
esac
LOG="results/logs/solve_tanh__$TAG.log"
echo "== $DEPTH: arch $ARCH, GPU $GPU, tag $TAG (log $LOG)"
python -u solve_tanh.py --arch "$ARCH" --tag "$TAG" --gpus "$GPU" $OPTS $S 2>&1 | tee "$LOG" | grep -E "^\[solve_tanh\] == |final net"

if [ "$DEPTH" = "5L" ] || [ "$DEPTH" = "6L" ]; then
  # extra 8-step all-free K-FAC polish (512 points x 64 directions, +131,584 queries), as in the record
  AT=$(echo "$ARCH" | tr ',' 'x')
  POP="recon/_pop__mergedbest512_tanh_${TAG}__${AT}__s0.pt"
  echo "== extra all-free polish -> recon/_final2__$TAG.pt (log results/logs/final2__$TAG.log)"
  PRECOND=kfac CUDA_VISIBLE_DEVICES="$GPU" python -u peel/polish_full.py "$POP" "recon/_final__$TAG.pt" \
    "recon/_final2__$TAG.pt" 512 64 8 2000 2>&1 | tee "results/logs/final2__$TAG.log" | grep -E "before:|after:"
fi

if [ "$SUBDEPTH" = 1 ]; then
  rm -f recon/mergedbest512_tanh_${TAG}_sub__200x*__s0.pt       # the committee's query-window dump (not used below)
  # extra polish of the solved net (recon/_final__$TAG.pt); recon/_guess__$TAG.pt only supplies the black box
  NL=$(( ${DEPTH%L} - 1 ))
  FRZ=$(python -c "print(','.join(f'layers.{i}.{p}' for i in range(1, $NL) for p in ('weight', 'bias')))")
  echo "== extra polish A: all free, subspace directions, 8 steps -> recon/_xpol_sub__$TAG.pt (log results/logs/xpol_sub__$TAG.log)"
  SUBSPACE_U="recon/_subspaceU__$TAG.pt" PRECOND=kfac CUDA_VISIBLE_DEVICES="$GPU" python -u peel/polish_full.py \
    "recon/_guess__$TAG.pt" "recon/_final__$TAG.pt" "recon/_xpol_sub__$TAG.pt" 512 64 8 4000 \
    2>&1 | tee "results/logs/xpol_sub__$TAG.log" | grep -E "before:|after:" | sed 's/| mean per-matrix.*//'
  echo "== extra polish B: layer 1 only, full-space directions, 4 steps -> recon/_xpol__$TAG.pt (log results/logs/xpol__$TAG.log)"
  FREEZE="$FRZ" PRECOND=kfac CUDA_VISIBLE_DEVICES="$GPU" python -u peel/polish_full.py \
    "recon/_guess__$TAG.pt" "recon/_xpol_sub__$TAG.pt" "recon/_xpol__$TAG.pt" 512 64 4 4000 \
    2>&1 | tee "results/logs/xpol__$TAG.log" | grep -E "after:" | sed 's/frozen=([^)]*)/frozen=(L2..)/; s/| mean per-matrix.*//'
  echo "== final net: recon/_xpol__$TAG.pt"
fi
echo "== done ($DEPTH)"
