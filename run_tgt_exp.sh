#!/bin/bash
# 创新②(靶点FiLM调制)救活验证：三方对比，全在硬负样本 + 5折 + 诚实评测下。
# 前提：先跑 python build_target_features.py 生成 ESM⊕角色 真特征。
#
#   real      = 完整模型 + 做实靶点特征(ESM蛋白向量⊕onco/TSG角色)
#   learnable = 完整模型 + 旧版可学习ID向量(之前证明FiLM在这上面没用)
#   no_film   = 干脆去掉FiLM
# 判读：real 明显 > learnable ≈ no_film  → 创新②被真特征救活。
#       real ≈ learnable ≈ no_film        → FiLM是结构问题，真特征也救不了。
set -e
EPOCHS=${1:-200}
mkdir -p logs

echo "===== A. real (做实靶点) ====="
python main.py --neg degree --tgtfeat real     --folds 5 --epochs $EPOCHS 2>&1 | tee logs/tgt_real.log
echo "===== B. learnable (旧可学习) ====="
python main.py --neg degree --tgtfeat learnable --folds 5 --epochs $EPOCHS 2>&1 | tee logs/tgt_learnable.log
echo "===== C. no_film (无FiLM) ====="
python main.py --neg degree --ablation no_film  --folds 5 --epochs $EPOCHS 2>&1 | tee logs/tgt_nofilm.log

echo "完成。比三份日志末尾的 AUC/AUPR/MCC/relACC：real 是否压过另外两个。"
