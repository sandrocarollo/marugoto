import os
import pandas as pd
import argparse
import subprocess
import shutil
import re

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
    wsi_name_clean = wsi_name.replace('.ndpi', '')  # Remove extension
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

def prepare_h5(args, all_h5_dir, riley_h5_dir):
    wsi_name_clean = args.wsi_name.replace('.ndpi', '')
    if args.score_type == 'riley':
        os.makedirs(riley_h5_dir, exist_ok=True)
        src = os.path.join(all_h5_dir, f"{wsi_name_clean}_class_tokens.h5")
        dst = os.path.join(riley_h5_dir, f"{wsi_name_clean}.h5")
        shutil.copy(src, dst)
        return riley_h5_dir
    else:
        return all_h5_dir

def get_scaling_factors(patient_id):
    if patient_id == "2019-162609_colon transverse_HE.1":
        return 0.997, 0.923
    if patient_id == "2019-129947_ileum_HE.1":
        return 0.917, 0.993
    return 1.0, 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_name', type=str, help="WSI filename (e.g., TRR241-B-Re-2019-162609_col_transv_-_2023-11-08_13.55.28.ndpi)")
    parser.add_argument('--score_type', type=str, choices=["riley", "cortina"], help="Score type: riley or cortina")
    parser.add_argument('--superimpose', action='store_true', help="Whether to superimpose heatmap")
    parser.add_argument('--threshold_map', type=float, default=0.0, help="Threshold for heatmap generation")
    args = parser.parse_args()

    # Paths
    slide_csv_path = "./results/metadata_HE/slide.csv"
    if args.score_type == 'riley':
        folds_base_dir = "./results/IBD_reg_riley_Virchow2_Berlin_mil_class_token"
        all_h5_dir = "./results/IBD_features_Virchow2/STAMP_macenko_virchow2"
    if args.score_type == 'cortina':
        folds_base_dir = "./results/IBD_reg_cortina_UNI2_Berlin_mil"
        all_h5_dir = "./results/IBD_features_UNI2/STAMP_macenko_mahmood-uni2"
    wsis_base_dir = "./data/imaging"
    output_dir = "./results/heatmaps"

    # Find patient ID
    patient_id = find_patient_id(args.wsi_name, slide_csv_path)
    print(f"[-] Found patient ID: {patient_id}")

    # Find fold
    fold_num = find_fold(patient_id, folds_base_dir)
    print(f"[-] Found fold: {fold_num}")

    # Paths
    train_dir = os.path.join(folds_base_dir, f"fold-{fold_num}")
    h5_dir_to_use = prepare_h5(args, all_h5_dir=all_h5_dir, riley_h5_dir="./results/temp_riley_h5")

    # WSI path
    wsi_path = find_wsi_path(args.wsi_name, wsis_base_dir)
    print(f"[-] Found WSI path: {wsi_path}")

    # Scale factors
    scale_x, scale_y = get_scaling_factors(patient_id)

    # Create output folder if needed
    final_output_dir = f"{output_dir}/{args.score_type}_{patient_id}"
    os.makedirs(final_output_dir, exist_ok=True)

    # Run the heatmap script
    cmd = [
        "python", "external_tools/marugoto/marugoto/visualizations/mil_heatmaps.py",
        "--train_dir", train_dir,
        "--h5_feature_dir", h5_dir_to_use,
        "--out_dir", final_output_dir,
        "--ws_path", wsi_path,
        "--alpha", "0.6",
        "--threshold_map", f"{args.threshold_map}",
        "--heatmap_scale_x", str(scale_x),
        "--heatmap_scale_y", str(scale_y)
    ]
    if args.superimpose:
        cmd.append("--superimpose")
    print(f"[-] Running command:\n{' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
