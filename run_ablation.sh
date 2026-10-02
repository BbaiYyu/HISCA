#!/bin/bash
# 消融实验（诚实评测：验证集选轮 + 可选硬负样本/冷启动）。
# 每个消融 = 5折 × 最多 EPOCHS 轮（验证AUC早停会提前结束）。
#
# 决定性实验——硬负样本下跑6个消融（推荐，这才能看出组件有没有用）:
#   bash run_ablation.sh                     # 默认 neg=degree, 200轮上限
# 对照——随机负样本(会饱和到0.97, 仅供对比):
#   bash run_ablation.sh random
# 后台:
#   nohup bash run_ablation.sh > logs/ablation_all.log 2>&1 &
#   tail -f logs/ablation_all.log
set -e
NEG=${1:-degree}          # degree=硬负样本(默认) / random=随机
EPOCHS=${2:-200}
PATIENCE=${3:-20}
mkdir -p logs

for AB in none no_film no_hyper gcn no_relhead no_graph; do
  echo "========== 消融: $AB  (neg=$NEG) =========="
  python main.py --ablation $AB --neg $NEG --folds 5 --epochs $EPOCHS --patience $PATIENCE \
      2>&1 | tee logs/ablation_${AB}_${NEG}.log
done

echo "全部消融完成。各自末尾 'k 折结果' 即最终指标。"
echo "看 full(none) 是否在 AUPR/MCC/relACC 上压过其它消融——压过=组件成立。"
