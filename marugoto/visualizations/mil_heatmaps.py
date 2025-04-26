# %%
from enum import Enum, auto
from typing import Mapping, Optional, Sequence, Tuple, List
from fastai.vision.learner import load_learner
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from marugoto.mil.data import get_target_enc
from matplotlib.patches import Patch
from scipy import interpolate
import torch
import openslide
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
from PIL import Image
import h5py
import argparse
import re

__all__ = ['plot_heatmaps_', 'MapType']
# list of allowed formats for whole slide images
wsi_suffixes = ['.svs', '.ndpi', '.tif', '.qptiff']
# define colours for heatmap plots here
colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
# colors = plt.cm.plasma(np.linspace(0, 1, 256))[:, :3]

class MapType(Enum):
    ATTENTION = auto()
    # PROBABILITY = auto() old classification
    PREDICTION = auto()
    CONTRIBUTION = auto()
"""
*ATTENTION* Heatmap:
What it shows: The relative importance or "attention" the model gives to each patch. High attention means the model focuses more on that patch when making predictions.
Usage: Useful to understand which areas of an input contribute most to the model's decision, independent of the actual output values.

*PREDICTION* Heatmap:
What it shows: The direct output of the regression model. This represents the predicted values for each patch, which in a regression context could be any continuous value.
Usage: Useful to visualize the actual predictions made by the model across different patches. This helps in identifying patterns in predictions over the input space.

*CONTRIBUTION* Heatmap:
What it shows: A combination of attention and prediction. It represents the contribution of each patch to the final prediction, weighted by how much attention the model gives to each patch.
Usage: Provides a nuanced view combining both attention and regression output, showing not just which patches the model focuses on, but how much they contribute to the overall prediction.
"""

def get_dict_maptype_to_coords_scores(h5_feature_path: Path, model: nn.Module,
                                      map_types: List[MapType] = [
                                          MapType.ATTENTION],
                                      ) -> dict:

    dict_maptype_to_coords_scores = {}

    feats, coords, sizes = [], [], []
    with h5py.File(h5_feature_path, 'r') as f:
        feats.append(torch.from_numpy(f['feats'][:]).float())
        sizes.append(len(f['feats']))
        coords.append(torch.from_numpy(f['coords'][:]))
    feats, coords = torch.cat(feats), torch.cat(coords)

    encoder = model.encoder.eval()
    attention = model.attention.eval()
    head = model.head.eval()

    # calculate attention, scores etc.
    encs = encoder(feats)
    patient_atts = torch.softmax(attention(encs), dim=0).detach()
    # patient_scores = torch.softmax(head(encs), dim=1).detach() old for classification
    patient_scores = head(encs).detach()
    normed_patient_atts=(patient_atts-patient_atts.min())/(patient_atts.max()-patient_atts.min())
    patient_weighted_scores=normed_patient_atts*patient_scores

    # classification assertion
    # assert patient_scores.shape[-1] <= colors.shape[0], f'not enough colours.\n'\
    #     'Can only plot score for max {colors.shape[0]}'\
    #     'classes at a time!\n Number of classes asked for:'\
    #     f'{len()} not supported.'

    for map_type in map_types:
        if map_type == MapType.ATTENTION:
            scores = patient_atts.numpy()
            scores -= scores.min()
            scores /= (scores.max() - scores.min())
        elif map_type == MapType.PREDICTION:
            scores = patient_scores.numpy()
        elif map_type == MapType.CONTRIBUTION:
            scores = patient_weighted_scores.numpy()
        else:
            raise ValueError(f'Heat map type {map_type} not supported!')

        dict_maptype_to_coords_scores[map_type] = coords.numpy(), scores

    return dict_maptype_to_coords_scores


def _MIL_heatmap_for_slide(coords: np.ndarray, scores: np.ndarray,
                           colours: np.ndarray = None, threshold_map: float = 1.0) -> np.ndarray:
    """
    Args: 
        h5_feature_path: path to .h5 file with features to analyse
        model: model to analyse slide with
        categories: TODO
        map_type: one from ['attention','probability', contribution]

    Returns:
        Tuple of covered_area, legend, heatmap
        covered_area: extent in x dimension and extent in y dimension of whole slide image
        legend: details of legend for plot
        heatmap: the actual heatmap, z coordinate is activation, x and y are integers from 0 to n_x
            and 0 to n_y, respectively. To get pixel dimensions multiply x and y by stride
    """

    # get stride
    stride = _get_stride(coords)
    scaled_map_coords = coords // stride
    if colours is not None:
        pass
    else:
        colours = colors

    # make a mask, 1 where coordinates have attention 0 otherwise
    # ndarray of zeros of dimension max_x * max_y
    mask = np.zeros(scaled_map_coords.max(0) + 1)
    # add in ones where we have values
    for coord in scaled_map_coords:
        mask[coord[0], coord[1]] = 1

    grid_x, grid_y = np.mgrid[0:scaled_map_coords[:, 0].max()+1,
                              0:scaled_map_coords[:, 1].max()+1]

    # interpolate heatmap over grid
    if scores.ndim < 2:
        scores = np.expand_dims(scores, 1)
    activations = interpolate.griddata(
        scaled_map_coords, scores, (grid_x, grid_y))
    activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

    heatmap = _visualize_activation_map(
        activations.transpose(1, 0, 2), colours[:activations.shape[-1]], threshold_map=threshold_map)

    return heatmap


def _plot_heatmap_(coords, heatmap,
                   outdir: Path, wsi_path: Optional[Path] = None,
                   superimpose: bool = True, alpha: float = 0.5,
                   heatmap_scale_x: float = 1.0, heatmap_scale_y: float = 1.0) -> None:
    format = '.svg'
    stride = _get_stride(coords)
    covered_area = (coords.max(0) + stride)
    plt.figure(dpi=600)
    if wsi_path:
        from openslide import OpenSlide

        assert wsi_path.suffix in wsi_suffixes, \
            f'Cannot read files with extension {wsi_path.suffix}. ' \
            f'Please provide a WSI with extension in {wsi_suffixes}.'
        title = wsi_path.stem if wsi_path else "Heatmap"
        slide = OpenSlide(str(wsi_path))
        level = next((i for i, dims in enumerate(slide.level_dimensions)
                      if max(dims) <= 2400*2), slide.level_count - 1)
        thumb = slide.read_region((0, 0), level, slide.level_dimensions[level])
        thumb = thumb.convert("RGBA")  # Ensure RGBA mode for alpha compositing

        covered_area_size = (covered_area / slide.level_downsamples[level]).astype(int)
        covered_area_size = np.array(thumb.size)

        # Resize heatmap to match the thumbnail size
        heatmap = Image.fromarray(heatmap)
        heatmap_resized = heatmap.resize((int(covered_area_size[0] * heatmap_scale_x), 
                                          int(covered_area_size[1] * heatmap_scale_y)), 
                                         resample=Image.Resampling.NEAREST)
        
        if superimpose:
            # Apply transparency directly to heatmap alpha channel
            alpha_channel = heatmap_resized.split()[-1]
            alpha_channel = alpha_channel.point(lambda p: p * alpha)
            heatmap_resized.putalpha(alpha_channel)
            
            # Position the resized heatmap on top of the original thumbnail
            heatmap_position = ((thumb.width - heatmap_resized.width) // 2,
                                (thumb.height - heatmap_resized.height) // 2)
            heatmap_position = (0,0)
            # Create a new image with transparent background and paste both images
            combined_image = Image.new('RGBA', thumb.size, (255, 255, 255, 0))
            combined_image.paste(thumb, (0, 0))
            combined_image.paste(heatmap_resized, heatmap_position, mask=heatmap_resized)
            
            plt.figure(figsize=(12, 6), dpi=300)
            plt.imshow(combined_image)
            plt.axis('off')
            plt.title(wsi_path.stem)
        else:
            # Plotting side by side
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
            axs[0].imshow(thumb)
            axs[0].axis('off')
            axs[1].imshow(heatmap_resized)
            axs[1].axis('off')
            axs[1].set_title(wsi_path.stem)
    else:
        print(f'No path to WSI given, plotting heatmap without WSI ...\n')
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(np.multiply(heatmap.size, 8), resample=Image.Resampling.NEAREST)
        plt.figure(figsize=(12, 6), dpi=300)
        plt.imshow(heatmap)
        plt.axis('off')
        plt.title("Heatmap")  # Title when only heatmap is present    

    # Add legend to the figure
    # legend = plt.legend(title=title, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

    # Create a color bar for the attention values
    # Create a color bar for the attention values, considering alpha
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize based on heatmap range
    cmap = plt.get_cmap('viridis')
    cmap_with_alpha = cmap(np.linspace(0, 1, 256))
    cmap_with_alpha[:, -1] = alpha  # Apply the same alpha transparency
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(cmap_with_alpha), norm=norm)
    sm.set_array([])

    cbar_obj = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
    cbar_obj.set_label('Attention Level')

    out_file = (outdir / wsi_path.stem).with_suffix(format) if wsi_path else (outdir / "heatmap").with_suffix(format)
    print('[HEATMAP]')
    print(f'Writing output to file: {out_file}')
    out_file.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(out_file, bbox_inches='tight')
    plt.close('all')


def _get_stride(coordinates: np.ndarray) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)

    return stride


def _visualize_activation_map(activations: np.ndarray, colors: np.ndarray, alpha: float = 1.,
    clipping: bool=True, threshold_map: float = 1.0) -> np.ndarray:
    """Transforms an activation map into an RGBA numpy array for regression tasks.
    Args:
        activations: An (h, w, 1) array of activations.
        colors: A (256, 3) array mapping each of the target classes to a color.
        alpha: Transparency level for the heaatmap
        clipping: Whether to clip the RGB values to prevent overflow
    Returns:
        An interpolated activation map. Regions which the algorithm assumes to be background
        will be transparent.
    """
    assert colors.shape[1] == 3, "expected color map to have three color channels"
    assert colors.shape[0] == activations.shape[2], "one color map entry per class required"
    # activations should be less or equal to 1
    assert activations[2].max() <= 1, f"Activations should be less than one, otherwise maps get clipped! \n \
        Max value provided {activations[2].max()}."
    
    norm_activations = np.clip(activations.squeeze(), 0, 1)  # Squeeze to remove the extra dimension if exists
    #print(f"Activations range after clipping: {norm_activations.min()} to {norm_activations.max()}")
    # Use the viridis colormap from matplotlib for more varied color mapping
    colormap = plt.cm.viridis
    rgbmap = colormap(norm_activations)[:, :, :3]  # Use only RGB, ignore alpha channel from colormap
    #print(f"Sample RGB values: {rgbmap[0, 0]}, {rgbmap[25, 25]}, {rgbmap[-1, -1]}")
    # Apply clipping if necessary
    if clipping:
        rgbmap = np.clip(rgbmap, 0, 1)  # Clip to make sure RGB values are within valid range
        
    # Create alpha channel
    # alpha_channel = (norm_activations * alpha).astype(np.float32)
    alpha_channel = np.ones_like(norm_activations) * alpha  # Constant alpha, no transparency
    alpha_channel[norm_activations <= threshold_map] = 0  # Make alpha 0 where activations are 0
    #print(f"Alpha channel range: {alpha_channel.min()} to {alpha_channel.max()}")

    # Stack RGB and alpha channels to create RGBA image
    im_data = np.dstack((rgbmap, alpha_channel))

    # Convert to 8-bit per channel
    im_data = (im_data * 255).astype(np.uint8)
    #print(f"Image data range after conversion to 8-bit: {im_data.min()} to {im_data.max()}")

    return im_data


def _get_slide_features(h5_feature_dir, ws_path):

    assert h5_feature_dir.is_dir(), \
        f'{h5_feature_dir} is not a directory. Please provide path to feature directory!'

    whole_slides = []
    h5_feature_paths = []

    if ws_path.is_file():
        whole_slides.append(ws_path)
        h5_feature_path = h5_feature_dir/ws_path.with_suffix('.h5').name
        h5_feature_paths.append(h5_feature_path)
    elif ws_path.is_dir():
        for suffix in wsi_suffixes:
            for ws_p in ws_path.glob(f'*{suffix}'):
                h5_feature_path = h5_feature_dir/ws_p.with_suffix('.h5').name
                if h5_feature_path.is_file():
                    whole_slides.append(ws_p)
                    h5_feature_paths.append(h5_feature_path)
                else:
                    print(
                        f'Could not find file {h5_feature_path}.\
                             Check if features where extracted for this whole slide image!')
    else:
        raise ValueError(
            f'Given ws_path is neither a file nor a directory. Path given {ws_path=!r}.')

    return list(zip(whole_slides, h5_feature_paths))

def save_top_patches_to_csv(dict_maptype_to_coords_scores, map_type, top_n):
    # Extract coordinates and scores
    coords, scores = dict_maptype_to_coords_scores[map_type]
    
    # Flatten scores to 1D if necessary (for multi-class cases)
    scores = scores.squeeze()

    # Combine coords and scores for sorting
    data = list(zip(coords, scores))
    
    # Sort by score in descending order (highest score first)
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    
    # Get the top N patches
    top_patches = sorted_data[:top_n]
    
    # Create a DataFrame for easy saving
    df = pd.DataFrame({
        'coords': [f"({int(x[0])}, {int(x[1])})" for x, _ in top_patches],
        'attention_score': [score for _, score in top_patches]
    })
    return df

def extract_positions(coords):
    # Use regex to find numbers in the string
    match = re.findall(r'\d+', coords)
    if match:
        pos_0 = int(match[0])
        pos_1 = int(match[1])
        return pos_0, pos_1
    return None, None

def get_n_toptiles(
    slide_path,
    output_dir,
    scores,
    stride: int,
    n: int = 8,
    tile_size: int = 224,
    thumbnail_size: tuple = (2048, 2048)
) -> None:
    slide = openslide.open_slide(slide_path)
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])

    # determine the scaling factor between heatmap and original slide
    # 256 microns edge length by default, with 224px = ~1.14 MPP (± 10x magnification)
    feature_downsample_mpp = (
        256 / stride
    )  # NOTE: stride here only makes sense if the tiles were NON-OVERLAPPING
    scaling_factor = feature_downsample_mpp / slide_mpp

    top_score = scores.head(n).reset_index()

    # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
    # then add [top_score.values > 0.5]
    for index, row in top_score.iterrows():
        # Extract positions from the score row
        pos_0, pos_1 = extract_positions(row['coords'])
        
        # Ensure positions are valid
        if pos_0 is None or pos_1 is None:
            print(f"[ERROR] Invalid coordinates: {row['coords']}")
            continue
        # print(f"Scaling factor: {scaling_factor}")
        # Scale positions to match slide resolution
        scaled_pos_0 = int(pos_0 * scaling_factor)
        scaled_pos_1 = int(pos_1 * scaling_factor)

        # Define target tile size
        target_size = (int(1 * stride * scaling_factor), int(1 * stride * scaling_factor))

        # Read the region from the slide
        tile = (
            slide.read_region(
                (scaled_pos_0, scaled_pos_1),
                0,
                target_size
            )
            .convert("RGB")
            .resize((tile_size, tile_size))  # Resize to desired tile size
        )

        # Construct filename and save path
        tile_filename = f"toptiles_{index+1}_({pos_0},{pos_1}).jpg"
        tile_output_dir = output_dir / "toptiles"
        tile_output_dir.mkdir(exist_ok=True, parents=True)
        tile_path = tile_output_dir / tile_filename

        # Save the tile
        try:
            tile.save(tile_path)
            print(f"Tile saved at {tile_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save tile at {tile_path}: {e}")
        
    # Create and save a thumbnail of the entire slide
    thumbnail = slide.get_thumbnail(thumbnail_size)  # Resize the entire slide to desired size
    thumbnail_filename = "slide_thumbnail.jpg"
    thumbnail_path = output_dir / thumbnail_filename
    thumbnail.save(thumbnail_path)
    print('[THUMBNAIL]')
    print(f"Thumbnail saved at {thumbnail_path}")

def plot_heatmaps_(out_dir: Path, train_dir: Path, ws_path: Path, h5_feature_dir: Path,
                  map_types: List[MapType] = [MapType.ATTENTION],
                  superimpose: bool = True, alpha: float = 0.5, threshold_map: float = 1.0,
                  heatmap_scale_x: float = 1.0, heatmap_scale_y: float = 1.0):
    """Generates heatmaps for whole slide images.

    Outputs heatmaps to project directory, in subfolders for each map_type.

    Args:
        out_dir: path to where outputs are stored
        train_dir: path to directory where training was done, i.e. where export.pkl file is located
        ws_path: path to whole slide image, either full path -> single image is analysed or directory 
            -> all whole slide images in directory are analysed
        h5_feature_dir: directory containing features used in training, must match whole slide images
        map_types: list containing attention, probability and/or contribution to give corresponding heatmaps
        superimpose: have heatmap on top of thumbnail or both side-by-side
        alpha: transparacy of heatmap
    """
    slide_features = _get_slide_features(h5_feature_dir, ws_path)
    learn = load_learner(train_dir/'export.pkl')
    # DEBUG
    # print(type(learn.dls.train.dataset))
    # print(learn.dls.train.dataset)
    # print(dir(learn.dls.train.dataset))
    # print(learn.model)
    
    # print(learn.dls.train.dataset._datasets[-1])

    # target_enc = get_target_enc(learn)
    target_enc = learn.dls.train.dataset._datasets[-1][0]
    categories = np.unique(target_enc)

    str_targets = ['contrib_'+np.array2string(target) for target in categories]

    for slide_path, h5_feature_path in slide_features:
        dict_maptype_to_coords_scores = get_dict_maptype_to_coords_scores(h5_feature_path,
                                                                          model=learn.model, map_types=map_types)
        
        for map_type in map_types:
            coords, scores = dict_maptype_to_coords_scores[map_type]
            if map_type == MapType.ATTENTION:
                legend_elements = [
                    Patch(facecolor=colors[0], label='attention')]
            else:
                legend_elements = [Patch(facecolor=color, label=class_) for class_,
                                   color in zip(str_targets, colors)]
            # print(f"Coordinates shape: {coords.shape}, range: {coords.min()} to {coords.max()}")
            # print(f"Scores shape: {scores.shape}, range: {scores.min()} to {scores.max()}")

            heatmap = _MIL_heatmap_for_slide(coords=coords, scores=scores, threshold_map=threshold_map)

            _plot_heatmap_(coords, heatmap=heatmap,
                           outdir=out_dir/map_type.name, wsi_path=slide_path, superimpose=superimpose, 
                           alpha=alpha, heatmap_scale_x=heatmap_scale_x, heatmap_scale_y=heatmap_scale_y)
            
            score_ranking_df = save_top_patches_to_csv(dict_maptype_to_coords_scores, map_type=map_type, top_n=50)
            out_file = ((out_dir/map_type.name) / "top_patches_ranking.csv")
            score_ranking_df.to_csv(out_file, index=True, index_label='')
            print('[TOP PATCHES]')
            print(f'Top patches saved in {out_file}')
            # Top tiles generation part:
            n_toptiles = 10
            print('[TOP TILES]')
            print(f"Generation of {n_toptiles} top tiles.")
            print(f"Creating top tiles...")
            stride = _get_stride(coords)
            get_n_toptiles(
                slide_path=slide_path,
                stride=stride,
                output_dir=out_dir/map_type.name,
                scores=score_ranking_df,
                n=n_toptiles,
                tile_size=512,
                thumbnail_size=(2048, 2048)
            )
            

def get_overlay(thumb,covered_area_size, coords, scores, alpha=0.6, colors=colors):
    """ takes a thumb image, resizes it to covered_area_size, gets heatmap for scores
        and overlays score heatmap over thumb image.
    """
# get attention map in overlay
    heatmap = _MIL_heatmap_for_slide(coords=coords, scores= scores,
                            colours=colors)
    heatmap[:, :, -1] = heatmap[:, :, -1]*alpha
    heatmap = Image.fromarray(heatmap)
    # make heatmap and thumb the same size
    scaled_heatmap = Image.new('RGBA', thumb.size)
    scaled_heatmap.paste(heatmap.resize(
        covered_area_size, resample=Image.Resampling.NEAREST))
    return Image.alpha_composite(thumb, scaled_heatmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heatmaps for whole slide images.")

    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to store output heatmaps.")
    parser.add_argument("--train_dir", type=Path, required=True, help="Directory where training was done (contains export.pkl).")
    parser.add_argument("--ws_path", type=Path, required=True, help="Path to a whole slide image or a directory of images.")
    parser.add_argument("--h5_feature_dir", type=Path, required=True, help="Directory containing the features used in training.")
    parser.add_argument("--map_types", type=str, nargs='*', default=['ATTENTION'], help="List of map types (ATTENTION, PREDICTION, CONTRIBUTION).")
    parser.add_argument("--superimpose", action="store_true", help="If present, superimpose the heatmap on the image (default: False)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency level of the heatmap overlay.")
    parser.add_argument("--threshold_map", type=float, default=1.0, help="Threshold of the heatmap to focus on high attention areas.")
    parser.add_argument("--heatmap_scale_x", type=float, default=1.0, help="Scaling factor for heatmap width (default: 1)")
    parser.add_argument("--heatmap_scale_y", type=float, default=1.0, help="Scaling factor for heatmap height (default: 1)")
    args = parser.parse_args()
    # Convert string map types to MapType enum values
    map_types = [MapType[mt] for mt in args.map_types]

    # Call the main function with parsed arguments
    plot_heatmaps_(
        out_dir=args.out_dir,
        train_dir=args.train_dir,
        ws_path=args.ws_path,
        h5_feature_dir=args.h5_feature_dir,
        map_types=map_types,
        superimpose=args.superimpose,
        alpha=args.alpha,
        threshold_map = args.threshold_map,
        heatmap_scale_x=args.heatmap_scale_x,
        heatmap_scale_y=args.heatmap_scale_y
    )
