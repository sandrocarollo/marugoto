#!/bin/bash

# Check if the argument is valid
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [riley|cortina]"
    exit 1
fi

TASK=$1

if [ "$TASK" == "riley" ]; then
    CLINI_EXCEL="./results/metadata_HE/clini_table_riley_berlin.csv"
    SLIDE_CSV="./results/metadata_HE/slide_class_token.csv"
    FEATURE_DIR="./results/IBD_features_Virchow2/STAMP_macenko_virchow2"
    TARGET_LABEL="normalized_riley_score"
    OUTPUT_PATH="./results/IBD_riley_Virchow2_Berlin_mil"
elif [ "$TASK" == "cortina" ]; then
    CLINI_EXCEL="./results/metadata_HE/clini_table_cortina_berlin.csv"
    SLIDE_CSV="./results/metadata_HE/slide.csv"
    FEATURE_DIR="./results/IBD_features_UNI2/STAMP_macenko_mahmood-uni2"
    TARGET_LABEL="normalized_naini_cortina_score"
    OUTPUT_PATH="./results/IBD_cortina_UNI2_Berlin_mil"
else
    echo "Invalid scoring system: $TASK"
    echo "Please choose 'riley' for Riley score or 'cortina' for Naini Cortina score"
    exit 1
fi

N_SPLITS=5

echo "Starting 5-Fold Cross Validation for $TASK..."

python -m marugoto.mil crossval \
    --clini_excel "$CLINI_EXCEL" \
    --slide_csv "$SLIDE_CSV" \
    --feature_dir "$FEATURE_DIR" \
    --target_label "$TARGET_LABEL" \
    --output_path "$OUTPUT_PATH" \
    --n_splits "$N_SPLITS"

echo "Cross-validation finished for $TASK."
