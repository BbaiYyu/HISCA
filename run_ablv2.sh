#!/bin/bash
# v2 真实留一消融 @ α=0.5 (5折). full=含HGT; 各变体去掉一个真实组件.
cd "D:/my_file/piRNA_new/ncdrug_model"
PY="D:/anaconda/envs/pytorch_gpu/python.exe"
export NCDRUG_HARD_FRAC=0.5
OUT="D:/my_file/piRNA_new/baselines"

run () {
  name="$1"; shift
  echo "########## RUN $name ##########"
  env "$@" "$PY" main.py --arch v2 --neg degree --folds 5 > "$OUT/ablv2_$name.log" 2>&1
  echo "== $name 结果 =="
  grep -aE "均值|mean|AUC" "$OUT/ablv2_$name.log" | tail -4
}

run full
run no_hgt   NCDRUG_NO_HGT=1
run no_kmer  NCDRUG_KMER_INT=0
run no_film  NCDRUG_NO_TYPEFILM=1
run no_mv    NCDRUG_NO_MULTIVIEW=1
echo "ALL_DONE_ABLV2"
