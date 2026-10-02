#!/bin/bash
# 优化A 快速版: 只比 baseline vs A_full, 各3折, 出"A到底涨不涨"的头条结论。
cd "D:/my_file/piRNA_new/ncdrug_model"
PY="D:/anaconda/envs/pytorch_gpu/python.exe"
export NCDRUG_HARD_FRAC=0.5
OUT="D:/my_file/piRNA_new/baselines"
mkdir -p "$OUT"

run () {
  name="$1"; shift
  echo "########## RUN $name ##########"
  env "$@" "$PY" main.py --arch v2 --neg degree --folds 3 > "$OUT/hgtAfast_$name.log" 2>&1
  echo "== $name 结果 =="
  grep -aE "均值|mean|MEAN|AUC.*AUPR|5折|3折|fold-mean" "$OUT/hgtAfast_$name.log" | tail -4
}

run baseline NCDRUG_HGT_JK=last NCDRUG_HGT_DROPEDGE=0   NCDRUG_HGT_LAYER=2
run A_full   NCDRUG_HGT_JK=cat  NCDRUG_HGT_DROPEDGE=0.2 NCDRUG_HGT_LAYER=3
echo "ALL_DONE_HGTA_FAST"
