import os
import re
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
from external_tools.marugoto.feature_vector_extractor_class import feat_ext_

# --- Logging Setup ---
log_path = "img_feature_extraction_classifier.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

def find_wsi_path(wsi_name, imaging_base_dir):
    match = re.search(r'TRR241-.*?-(\d{4})-', wsi_name)
    if match:
        year = match.group(1)
    else:
        raise ValueError(f"Could not extract year from WSI name: {wsi_name}")

    year_dir = os.path.join(imaging_base_dir, year)
    wsi_path = os.path.join(year_dir, wsi_name)
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")
    return wsi_path

def find_patient_id(wsi_name, slide_csv_path):
    for ext in ['.ndpi', '.qptiff', '.svs', '.tif']:
        if wsi_name.endswith(ext):
            wsi_name_clean = wsi_name.replace(ext, '')
            break
    slide_df = pd.read_csv(slide_csv_path)
    match = slide_df[slide_df['FILENAME'] == wsi_name_clean]
    if match.empty:
        raise ValueError(f"WSI {wsi_name_clean} not found in slide.csv!")
    return match['PATIENT'].values[0]

def find_fold(patient_id, folds_base_dir):
    for fold_num in range(5):  # Assuming 5 folds
        pred_csv = os.path.join(folds_base_dir, f"fold-{fold_num}", "patient-preds.csv")
        pred_df = pd.read_csv(pred_csv)
        if patient_id in pred_df['PATIENT'].values:
            return fold_num
    raise ValueError(f"Patient ID {patient_id} not found in any fold.")

def main():

    # Paths
    slide_csv_path = "./results/metadata_HE/slide.csv"
    wsis_base_dir = "./data/imaging"
    all_h5_dir = "./results/IBD_features_Virchow2"
    folds_base_dir = "./results/IBD_classifier_Virchow2_5fcv_mofa"
    output_dir = "./results/img_features"
    os.makedirs(output_dir, exist_ok=True)

    slide_df = pd.read_csv(slide_csv_path)
    filenames = slide_df['FILENAME'].tolist()

    out_lines = []
    log.info(f"[~] Processing {len(filenames)} WSIs...")

    for idx, wsi_base in enumerate(filenames):
        log.info(f"\n[{idx+1}/{len(filenames)}] Processing {wsi_base}")

        wsi_found = False
        for ext in [".ndpi", ".qptiff", ".svs", ".tif"]:
            try:
                wsi_name = wsi_base + ext
                ws_path = find_wsi_path(wsi_name, wsis_base_dir)

                patient_id = find_patient_id(wsi_name, slide_csv_path)
                fold_num = find_fold(patient_id, folds_base_dir)
                train_dir = os.path.join(folds_base_dir, f"fold-{fold_num}")
                output_subdir = f"{output_dir}/class_{patient_id}"
                os.makedirs(output_subdir, exist_ok=True)

                # Run feature extraction
                vector = feat_ext_(
                    out_dir=Path(output_subdir),
                    train_dir=Path(train_dir),
                    ws_path=Path(ws_path),
                    h5_feature_dir=Path(all_h5_dir),
                )
                flat_vector = vector.flatten()
                line = f"{wsi_name} " + " ".join(map(str, flat_vector.tolist()))
                out_lines.append(line)
                log.info(f"[✓] Features extracted for {patient_id}.")
                wsi_found = True   

            except Exception as e_ext:
                log.warning(f"[!] Could not process {wsi_base + ext}: {e_ext}")
                continue
            
            if wsi_found:
                break
        if not wsi_found:
            log.error(f"[x] Skipping {wsi_base}, no successful extraction.")

    # Save as CSV
    matrix_data = [line.split() for line in out_lines]
    df = pd.DataFrame(matrix_data)
    df.to_csv("./results/img_feature_matrix.csv", index=False, header=False)
    log.info(f"[✓] Final feature matrix saved in ./results/img_feature_matrix.csv, shape: {df.shape}")

if __name__ == "__main__":
    main()