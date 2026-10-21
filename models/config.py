

import os
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "models" else _HERE
DATA_DIR = os.environ.get("NCDRUG_DATA_DIR",
                          os.path.normpath(os.path.join(_ROOT, "data")))
FEAT_DIR = os.environ.get("NCDRUG_FEAT_DIR", os.path.join(_ROOT, "features"))
CKPT_DIR = os.environ.get("NCDRUG_CKPT_DIR", os.path.join(_ROOT, "ckpt"))


EDGE_CSV      = os.environ.get("NCDRUG_EDGE_CSV",
                               os.path.join(DATA_DIR, "merged_clean.csv"))
DRUG_CSV      = os.path.join(DATA_DIR, "drugs.csv")
NCRNA_CSV     = os.path.join(DATA_DIR, "ncRNA.csv")
DRUG_TGT_CSV  = os.path.join(DATA_DIR, "drug_targets.csv")


RELATIONS = ["resistant", "sensitive"]
NUM_REL   = len(RELATIONS)

CONFLICT_POLICY = "drop"


D_MODEL    = int(os.environ.get("NCDRUG_DMODEL", 256))
D_RNA_ENC  = 640
D_DRUG_ENC = 384


TGT_FEAT_MODE = "real"


RNA_BACKEND  = "rnafm"
DRUG_BACKEND = "chemberta"
CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
RNA_WINDOW   = 1022
RNA_MAX_LEN  = 4096
K_RNA_REGION = 32
LD_MAX       = 128
T_MAX        = 32
ENC_BATCH    = 8


SIM_K          = 10
HYPER_MIN_GROUP = 2

HYPER_USE_SIM_EGO = False
HYPER_GROUP_MAX   = 0
HYPER_MAX_EDGES   = 0


FILM_HIDDEN = 64
N_HGT_LAYER = int(os.environ.get("NCDRUG_HGT_LAYER", 2))
N_HGT_HEAD  = 2


N_HYPER_LAYER = 1
HEAD_HIDDEN = 64
DROPOUT     = float(os.environ.get('NCDRUG_DROPOUT', 0.3))


LR         = float(os.environ.get('NCDRUG_LR', 3e-4))
WEIGHT_DECAY = 5e-4
EPOCHS     = 200
K_FOLD     = 5
BATCH_SIZE = 2048
NEG_RATIO  = int(os.environ.get("NCDRUG_NEGRATIO", 1))
SEED       = 2024
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LAMBDA_REL = 1.0
