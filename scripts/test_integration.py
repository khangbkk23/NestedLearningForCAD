import os
# [MATH PUNCH]: Disable OpenMP conflict warnings on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from models.vit_cms import ViT_CMS 

def test_full_pipeline_zero_crash():
    print("--- STARTING ViT + TITANS INTEGRATION ANATOMY ---")
    
    # 1. Initialize model
    print("[Status] Initializing ViT_CMS (May take a few seconds to load weights if applicable)...")
    model = ViT_CMS(pretrained=False).cpu()
    model.train()
    
    # 2. Find the first CMS layer and use weights as a baseline
    cms_layer = next(m for m in model.modules() if m.__class__.__name__ == 'CMS')
    weight_before = cms_layer.fast_memory.net[0].weight.clone()

    # 3. Create dummy Data: Batch of 2 images (3 channels, 256x256)
    dummy_images = torch.randn(2, 3, 256, 256)

    # ==========================================
    # SCENARIO A: NORMAL IMAGE INGESTION
    # ==========================================
    print("\n[Status] Injecting NORMAL images into the system...")
    model.set_titans_learning_mode(is_normal_image=True) # OPEN VALVE
    
    out_normal = model(dummy_images)
    weight_after_normal = cms_layer.fast_memory.net[0].weight.clone()
    
    if not torch.equal(weight_before, weight_after_normal):
        print("[Pass] TITANS has SELF-LEARNED and successfully updated weights!")
    else:
        print("[CRASH] Weights frozen. update_allowed back-door is not working!")

    # ==========================================
    # SCENARIO B: ANOMALY IMAGE INGESTION
    # ==========================================
    print("\n[Status] Injecting ANOMALY images into the system...")
    model.set_titans_learning_mode(is_normal_image=False) # CLOSE VALVE
    
    out_anomaly = model(dummy_images)
    weight_after_anomaly = cms_layer.fast_memory.net[0].weight.clone()
    
    if torch.equal(weight_after_normal, weight_after_anomaly):
        print("[Pass] TITANS has FROZEN to protect memory. Zero-Leakage!")
    else:
        print("[CRASH] Knowledge leakage error! TITANS is still learning from anomaly images.")

if __name__ == "__main__":
    test_full_pipeline_zero_crash()