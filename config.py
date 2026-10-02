# -*- coding: utf-8 -*-
"""
全局配置：路径 + 超参数，集中在这里改。
对应模型框架图（模型框架.vsdx）里的 5 个阶段。
"""
import os
import torch

# ------------------------------------------------------------------ 路径
# 全部相对本文件位置，换机器/上服务器不用改代码。
# 如目录结构不同，可用环境变量覆盖：NCDRUG_DATA_DIR / NCDRUG_FEAT_DIR / NCDRUG_CKPT_DIR
_HERE = os.path.dirname(os.path.abspath(__file__))                       # = ncdrug_model/
DATA_DIR = os.environ.get("NCDRUG_DATA_DIR",
                          os.path.normpath(os.path.join(_HERE, "data")))  # 关联/序列 csv 放这
FEAT_DIR = os.environ.get("NCDRUG_FEAT_DIR", os.path.join(_HERE, "features"))  # 预计算特征 *.pt
CKPT_DIR = os.environ.get("NCDRUG_CKPT_DIR", os.path.join(_HERE, "ckpt"))

# 建模就绪的数据文件
EDGE_CSV      = os.environ.get("NCDRUG_EDGE_CSV",
                               os.path.join(DATA_DIR, "merged_clean.csv"))  # 关联表(可用环境变量换重平衡基准)
DRUG_CSV      = os.path.join(DATA_DIR, "drugs.csv")          # 药物 + SMILES
NCRNA_CSV     = os.path.join(DATA_DIR, "ncRNA.csv")          # ncRNA + Sequence
DRUG_TGT_CSV  = os.path.join(DATA_DIR, "drug_targets.csv")   # 药物→蛋白靶点 ID（★还需构建，缺失则自动降级）

# ------------------------------------------------------------------ 关系类型（创新①：分关系）
# 靶向(target)已删（实为表达变化、非真靶向），只保留两个真实功能关系
RELATIONS = ["resistant", "sensitive"]
NUM_REL   = len(RELATIONS)
# 耐药↔敏感冲突对怎么处理: "drop"=整个删掉  "mask"=仍算关联正样本但不做关系分类
CONFLICT_POLICY = "drop"

# ------------------------------------------------------------------ 维度
D_MODEL    = int(os.environ.get("NCDRUG_DMODEL", 256))  # 全模型统一隐藏维(128→256: 容量提升,实测+1.2点关联AUC)
D_RNA_ENC  = 640      # RNA-FM 每个核苷酸输出维（阶段1，若用别的编码器改这里）
D_DRUG_ENC = 384      # 药物编码器每个子结构输出维（如 ChemBERTa）

# ------------------------------------------------------------------ 靶点特征（创新②的输入）
# "real"=用 build_target_features.py 做好的 ESM蛋白向量⊕onco/TSG角色（做实靶点）
# "learnable"=旧版可学习ID向量（消融对照，之前证明FiLM在这上面没用）
TGT_FEAT_MODE = "real"

# ------------------------------------------------------------------ 阶段1 编码器
RNA_BACKEND  = "rnafm"      # 'rnafm'(需 pip install rna-fm) 或 'fallback'(离线占位)
DRUG_BACKEND = "chemberta"  # 'chemberta'(transformers) 或 'fallback'
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"   # hidden=384，与 D_DRUG_ENC 一致
RNA_WINDOW   = 1022         # RNA-FM 单窗口长度（模型位置上限1026，含cls/eos→序列≤1023，取1022留余量）
RNA_MAX_LEN  = 4096         # 单条 RNA 总长上限（=最多4个窗口；只对超长circRNA兜底，可调大）
K_RNA_REGION = 32           # 每条 RNA 池化成 K 个"区域"（= 交互图 M 的行数）
LD_MAX       = 128          # 药物 SMILES token 上限（M 的列数上限）
T_MAX        = 32           # 每个药保留的靶点数上限（不池化，超出截断）
ENC_BATCH    = 8            # 预计算时编码器 batch（按窗口计，8GB 显存友好）

# ------------------------------------------------------------------ 图 / 超图构建
SIM_K          = 10    # 相似性网络：每个节点连 top-K 个最相似邻居（余弦）
HYPER_MIN_GROUP = 2    # 共享靶点超边：组内至少多少个药才成一条超边
# —— 超图防过拟合（相似ego超边与HGT的sim边重复且爆量9000+，默认关掉，只留共享靶点高阶群）
HYPER_USE_SIM_EGO = False  # True 才加"每节点+相似邻居"的ego超边（冗余，易过拟合）
HYPER_GROUP_MAX   = 0      # 共享靶点组大小上限：>该值的超大hub组丢弃（0=不限）
HYPER_MAX_EDGES   = 0      # 超边总数上限：按组大小降序保留前N条（0=不限）← 你说的"选前多少条"

# ------------------------------------------------------------------ 各阶段结构超参（已整体缩小治过拟合）
FILM_HIDDEN = 64      # 阶段2 靶点路由 FiLM 里两个小网络的隐藏维（128→64）（创新②）
N_HGT_LAYER = int(os.environ.get("NCDRUG_HGT_LAYER", 2))  # 阶段3 HGT 层数（残差+LN后可安全加深, 试 2/3/4）
N_HGT_HEAD  = 2       # 阶段3 HGT 注意力头数（4→2）
# —— HGT 升级版旋钮（在 layers.HGTBranch 内按 env 读取，默认即开）——
#   NCDRUG_HGT_JK       = cat(默认) | max | last   跨层特征聚合(JumpingKnowledge)
#   NCDRUG_HGT_DROPEDGE = 0.2(默认)                训练期随机丢弃的关联边比例(0=关, 抗"度捷径"过拟合)
N_HYPER_LAYER = 1     # 阶段3 超图层数（2→1，减一跳传播免过平滑/过拟合）
HEAD_HIDDEN = 64      # 阶段4 关系条件预测头隐藏维（128→64）（创新①）
DROPOUT     = float(os.environ.get('NCDRUG_DROPOUT', 0.3))     # 0.2→0.3 加大随机失活

# ------------------------------------------------------------------ 训练
LR         = float(os.environ.get('NCDRUG_LR', 3e-4))     # 1e-3→3e-4 降学习率，压住"3轮冲顶就崩"的过拟合，峰后移、下坡平缓
WEIGHT_DECAY = 5e-4   # 1e-4→5e-4 加大 L2 正则
EPOCHS     = 200
K_FOLD     = 5          # 5 折交叉验证
BATCH_SIZE = 2048       # 每步吃更多对，减少重算图次数（2048 对 8GB 更稳，避免显存顶满）
NEG_RATIO  = int(os.environ.get("NCDRUG_NEGRATIO", 1))        # 每个正样本配几个负样本
SEED       = 2024
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练总损失 = 是否关联(BCE) + λ·关系类型(CE)
LAMBDA_REL = 1.0
