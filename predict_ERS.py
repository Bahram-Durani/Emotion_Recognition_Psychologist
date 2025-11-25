# predict_ERS.py
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
# DO NOT MOVE ANYTHING ABOVE THIS LINE

import sys
import json
import random
import torch
import torch.nn.functional as F

sys.path.append(r"E:\Research_Datasets\Codes")
from ER_Sychologist import TrimodalEmotionModel, TrimodalDataset

CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
SPLIT_DIR  = r"E:\Research_Datasets\processed_features_split2\split_5"            # you can change the split number to split_1, slpilt_2, ... split_5 
MODEL_PATH = r"E:\Research_Datasets\trimodal_results_20251101_002153\split_5\best_model.pt"      # Here the split number (split_2) should match with the split directory 
ADVICE_JSON = r"E:\Research_Datasets\Codes\psychologist_rules.json"

with open(ADVICE_JSON, "r", encoding="utf-8") as f:
    PSYCH = json.load(f)

def get_advice(emotion_code: str):
    if "_meta" in PSYCH and "aliases" in PSYCH["_meta"]:
        aliases = PSYCH["_meta"]["aliases"]
        emotion_code = aliases.get(emotion_code, emotion_code)
    return PSYCH.get(emotion_code, {"message": "No advice found."})

def load_model(model_path: str, device: torch.device):
    model = TrimodalEmotionModel().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = TrimodalDataset(os.path.join(SPLIT_DIR, "train"))
    test_ds = TrimodalDataset(
        os.path.join(SPLIT_DIR, "test"),
        video_mean=train_ds.video_mean, video_std=train_ds.video_std,
        audio_mean=train_ds.audio_mean, audio_std=train_ds.audio_std,
        pose_mean=train_ds.pose_mean,   pose_std=train_ds.pose_std,
    )

    n = min(5, len(test_ds))
    idxs = random.sample(range(len(test_ds)), n)
    model = load_model(MODEL_PATH, device)

    results = []
    for i in idxs:
        sample_id = test_ds.samples[i]
        sample = test_ds[i]

        video = sample["video"].unsqueeze(0).to(device)
        audio = sample["audio"].unsqueeze(0).to(device)
        pose  = sample["pose"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(video, audio, pose, labels=None)
            logits = out["logits"]
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_code = CLASS_NAMES[pred_idx]
            pred_conf = float(probs[pred_idx].item())

        advice = get_advice(pred_code)

        print("=" * 70)
        print(f"Sample: {sample_id}")
        print(f"Predicted Emotion: {pred_code}  |  Confidence: {pred_conf:.3f}")

        for section_key, tips in advice.items():
            if section_key.startswith("_"):
                continue
            print(f"\n{section_key.replace('_', ' ').capitalize()}:")
            if isinstance(tips, list):
                for tip in tips[:4]:
                    print(f" - {tip}")
            else:
                print(f" - {tips}")

        results.append({
            "sample_id": sample_id,
            "pred_code": pred_code,
            "confidence": pred_conf,
        })

    out_json = os.path.join(os.path.dirname(MODEL_PATH), "random5_predictions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved summary:", out_json)

if __name__ == "__main__":
    main()

