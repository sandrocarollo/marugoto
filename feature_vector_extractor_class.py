# %%
from fastai.vision.learner import load_learner
import numpy as np
from marugoto.mil.data import get_target_enc
import torch
import pandas as pd
from pathlib import Path
import h5py
import argparse
import warnings
warnings.filterwarnings("ignore", message="load_learner` uses Python's insecure pickle module.*")

# list of allowed formats for whole slide images
wsi_suffixes = ['.svs', '.ndpi', '.tif', '.qptiff']

def _get_slide_features(h5_feature_dir, ws_path):

    assert h5_feature_dir.is_dir(), \
        f'{h5_feature_dir} is not a directory. Please provide path to feature directory!'

    whole_slides = []
    h5_feature_paths = []

    if ws_path.is_file():
        whole_slides.append(ws_path)
        h5_feature_path = h5_feature_dir / (ws_path.stem + '_class_tokens.h5')
        h5_feature_paths.append(h5_feature_path)
    else:
        raise ValueError(
            f'Given ws_path is neither a file nor a directory. Path given {ws_path=!r}.')

    return list(zip(whole_slides, h5_feature_paths))

def feat_ext_(out_dir: Path, train_dir: Path, ws_path: Path, h5_feature_dir: Path):
    """Generates feature vector for whole slide images.

    Args:
        out_dir: path to where outputs are stored
        train_dir: path to directory where training was done, i.e. where export.pkl file is located
        ws_path: path to whole slide image, either full path -> single image is analysed or directory 
            -> all whole slide images in directory are analysed
        h5_feature_dir: directory containing features used in training, must match whole slide images
    """
    # Load trained model
    slide_features = _get_slide_features(h5_feature_dir, ws_path)
    learn = load_learner(train_dir/'export.pkl')
    model = learn.model.eval()
    categories = learn.dls.train.dataset._datasets[-1].encode.categories_[0] 

    for slide_path, h5_feature_path in slide_features:
        # Load features
        with h5py.File(h5_feature_path) as f:
            feats = torch.tensor(f["feats"][:]).float()
            coords = torch.tensor(f["coords"][:], dtype=torch.int)

        # Get category scores and gradcam
        scores = torch.softmax(
            learn.model(feats.unsqueeze(-2), torch.ones((len(feats)))), dim=1)

        # Weighted top tiles from gradcam for that category
        category_idx = 1
        tile_scores = scores[:, category_idx]
        top_scores = tile_scores.topk(10)
        top_coords = coords[top_scores.indices]
        top_feats = feats[top_scores.indices]
        weights = top_scores.values.detach().cpu().numpy()
        weights = weights / weights.sum()

        weighted_feat = np.average(top_feats.detach().cpu().numpy(), axis=0, weights=weights)
        # Save top tile scores
        score_ranking_df = pd.DataFrame({
            "coords": [f"({int(c[0])}, {int(c[1])})" for c in top_coords.detach().cpu().numpy()],
            "score": top_scores.values.detach().cpu().numpy(),
        }).sort_values("score", ascending=False).reset_index(drop=True) 

        score_ranking_df.to_csv(out_dir / f"{ws_path.stem}_top_tiles.csv", index=False)
        print(f"[O] Feature matrix saved in {out_dir}/{ws_path.stem}_top_tiles.csv")
        print(f"[O] Feature vector extracted for {ws_path}, shape: {weighted_feat.shape}")
        return weighted_feat       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps for whole slide images.")

    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to store output heatmaps.")
    parser.add_argument("--train_dir", type=Path, required=True, help="Directory where training was done (contains export.pkl).")
    parser.add_argument("--ws_path", type=Path, required=True, help="Path to a whole slide image or a directory of images.")
    parser.add_argument("--h5_feature_dir", type=Path, required=True, help="Directory containing the features used in training.")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    feat_ext_(
        out_dir=args.out_dir,
        train_dir=args.train_dir,
        ws_path=args.ws_path,
        h5_feature_dir=args.h5_feature_dir
    )
