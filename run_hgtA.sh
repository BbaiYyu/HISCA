#!/bin/bash
# 优化A: HGT 加深(残差+LN)+JumpingKnowledge+DropEdge 的对照消融 @ 硬负采样5折。
# 逐个隔离 A 的三个部件, 看各自增益。跟 run_ablv2.sh 同协议(--arch v2 --neg degree)。
cd "D:/my_file/piRNA_new/ncdrug_model"
PY="D:/anaconda/envs/pytorch_gpu/python.exe"      # 服务器上换成对应 python
export NCDRUG_HARD_FRAC=0.5
OUT="D:/my_file/piRNA_new/baselines"
mkdir -p "$OUT"

run () {
  name="$1"; shift
  echo "########## RUN $name ##########"
  env "$@" "$PY" main.py --arch v2 --neg degree --folds 5 > "$OUT/hgtA_$name.log" 2>&1
  echo "== $name 结果 =="
  grep -aE "均值|mean|AUC" "$OUT/hgtA_$name.log" | tail -4
}

# 基线: 尽量还原论文版 HGT(JK=last, 不丢边, 2层) —— 残差/LN 现恒开, 作为公平参考
run baseline   NCDRUG_HGT_JK=last NCDRUG_HGT_DROPEDGE=0   NCDRUG_HGT_LAYER=2
# 完整 A: JK-cat + DropEdge0.2 + 3层
run A_full     NCDRUG_HGT_JK=cat  NCDRUG_HGT_DROPEDGE=0.2 NCDRUG_HGT_LAYER=3
# 隔离"加深"增益: A 但只 2 层
run A_L2       NCDRUG_HGT_JK=cat  NCDRUG_HGT_DROPEDGE=0.2 NCDRUG_HGT_LAYER=2
# 隔离"DropEdge"增益: A 但不丢边
run A_noDrop   NCDRUG_HGT_JK=cat  NCDRUG_HGT_DROPEDGE=0   NCDRUG_HGT_LAYER=3
# 更深看是否继续涨: A 但 4 层
run A_L4       NCDRUG_HGT_JK=cat  NCDRUG_HGT_DROPEDGE=0.2 NCDRUG_HGT_LAYER=4

echo "ALL_DONE_HGTA"
