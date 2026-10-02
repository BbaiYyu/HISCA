#!/bin/bash
# 留一消融 @ α=0.25 (度数匹配混合, 5折, 全指标). NO_HGT. 去掉一个内容组件.
cd "D:/my_file/piRNA_new/ncdrug_model"
PY="D:/anaconda/envs/pytorch_gpu/python.exe"
OUT="D:/my_file/piRNA_new/baselines"
run () { name="$1"; shift; env "$@" "$PY" main.py --arch v2 --neg degree --folds 5 --epochs 32 --patience 8 > "$OUT/loo_a025_$name.log" 2>&1; echo "done $name"; }
run nokmer NCDRUG_HARD_FRAC=0.25 NCDRUG_NO_HGT=1 NCDRUG_KMER_INT=0
run nofilm NCDRUG_HARD_FRAC=0.25 NCDRUG_NO_HGT=1 NCDRUG_NO_TYPEFILM=1
run nomv   NCDRUG_HARD_FRAC=0.25 NCDRUG_NO_HGT=1 NCDRUG_NO_MULTIVIEW=1
echo "LOO_A025_DONE"
