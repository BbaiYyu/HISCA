#!/bin/bash
# 留一消融 @ 随机负样本 (5折, 全指标). NO_HGT(HGT待用户模型). 去掉一个内容组件.
cd "D:/my_file/piRNA_new/ncdrug_model"
PY="D:/anaconda/envs/pytorch_gpu/python.exe"
OUT="D:/my_file/piRNA_new/baselines"
run () { name="$1"; shift; env "$@" "$PY" main.py --arch v2 --neg random --folds 5 --epochs 30 --patience 8 > "$OUT/loo_random_$name.log" 2>&1; echo "done $name"; }
run nokmer NCDRUG_NO_HGT=1 NCDRUG_KMER_INT=0
run nofilm NCDRUG_NO_HGT=1 NCDRUG_NO_TYPEFILM=1
run nomv   NCDRUG_NO_HGT=1 NCDRUG_NO_MULTIVIEW=1
echo "LOO_RANDOM_DONE"
