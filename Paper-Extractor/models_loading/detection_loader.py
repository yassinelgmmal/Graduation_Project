import os
import layoutparser as lp
from config import (
    DETECTRON2_CONFIG_PATH,
    DETECTRON2_MODEL_PATH,
    DEVICE,
    SCORE_THRESHOLD,
    LABEL_MAP
)

def load_model():
    # Check if files exist
    if not os.path.exists(DETECTRON2_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {DETECTRON2_CONFIG_PATH}")
    if not os.path.exists(DETECTRON2_MODEL_PATH):
        raise FileNotFoundError(f"Model weights file not found at {DETECTRON2_MODEL_PATH}")

    # Load layout model
    model = lp.models.Detectron2LayoutModel(
        config_path=DETECTRON2_CONFIG_PATH,
        model_path=DETECTRON2_MODEL_PATH,
        label_map=LABEL_MAP,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", SCORE_THRESHOLD],
        device=DEVICE
    )
    return model
