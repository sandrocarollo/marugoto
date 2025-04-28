#!/bin/bash

# Check if an argument has been provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [riley|cortina]"
    exit 1
fi

TASK=$1

if [ "$TASK" == "riley" ]; then
    CLINI_TABLE="./results/metadata_HE/clini_table_riley_erlangen.csv"
    SLIDE_CSV="./results/metadata_HE/slide_class_token.csv"
    FEATURE_DIR="./results/IBD_features_Virchow2/STAMP_macenko_virchow2"
    TARGET_LABEL="normalized_riley_score"
    MODEL_BASE_PATH="./results/IBD_riley_Virchow2_Berlin_mil"
    OUTPUT_BASE_PATH="./results/deploy_erlangen_riley_Virchow_mil"
elif [ "$TASK" == "cortina" ]; then
    CLINI_TABLE="./results/metadata_HE/clini_table_cortina_erlangen.csv"
    SLIDE_CSV="./results/metadata_HE/slide.csv"
    FEATURE_DIR="./results/IBD_features_UNI2/STAMP_macenko_mahmood-uni2"
    TARGET_LABEL="normalized_naini_cortina_score"
    MODEL_BASE_PATH="./results/IBD_cortina_UNI2_Berlin_mil"
    OUTPUT_BASE_PATH="./results/deploy_erlangen_cortina_UNI2_mil"
else
    echo "Invalid scoring system: $TASK"
    echo "Please choose 'riley' for Riley score or 'cortina' for Naini Cortina score"
    exit 1
fi

echo "Starting deployment to Erlangen cohort for $TASK..."

for fold in {0..4}; do
    echo "Deploying fold-$fold..."
    python -m marugoto.mil deploy \
        --clini_table "$CLINI_TABLE" \
        --slide_csv "$SLIDE_CSV" \
        --feature_dir "$FEATURE_DIR" \
        --target_label "$TARGET_LABEL" \
        --model_path "${MODEL_BASE_PATH}/fold-${fold}/export.pkl" \
        --output_path "${OUTPUT_BASE_PATH}_fold-${fold}"
done

echo "Deployment finished for $TASK."
