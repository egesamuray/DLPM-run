import os
import tqdm

import numpy as np
import torch
from PIL import Image
from pywt import dwt2, idwt2
import blobfile as bf
from torch.utils.data import DataLoader, Dataset


def load_data_wavelet(
    data_dir, 
    batch_size, 
    j, 
    conditional=True, 
    deterministic=False,
    debug=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs, where each
    'images' is an NCHW float tensor, and kwargs is a dict possibly containing
    the 'conditional' data.

    SINGLE-CHANNEL wavelet version:
      - We expect .npz files containing wavelet coeffs at scale j, shape (4, H', W').
      - If `conditional=True`, we yield (3, H', W') as the target (cH,cV,cD) and
        'conditional' => (1, H', W') as the low-freq subband (cA).
      - If not conditional, we yield cA alone.
    
    :param data_dir: a dataset directory containing .npz files (one per image).
    :param batch_size: how many samples per batch
    :param j: scale index to load, e.g. 'j=1' for the first scale
    :param conditional: whether to treat the cA subband as conditioning
    :param deterministic: if True, yields in a deterministic (shuffle=False) order
    :param debug: if True, prints debug messages once at the start of loading
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    from mpi4py import MPI

    # Collect all .npz files in 'data_dir' (recursively).
    all_files = _list_image_files_recursively(data_dir)
    if debug and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"[DEBUG] Found {len(all_files)} npz files in '{data_dir}' for scale j={j}")

    dataset = WaveletDataset(
        wav_paths=all_files,
        j=j,
        conditional=conditional,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        debug=debug
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(not deterministic),
        num_workers=2,   # can adjust as needed
        drop_last=True
    )
    loader = DataLoader(dataset, **loader_kwargs)

    # Return an infinite generator over the DataLoader
    while True:
        for batch in loader:
            yield batch


def _list_image_files_recursively(data_dir):
    """
    Look for .npz files in data_dir (and subdirectories),
    ignoring any file containing 'stats' in the name. 
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        # Only .npz are loaded:
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() == "npz" and "stats" not in entry:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


############################################
# The wavelet mean/std logic for SINGLE CHANNEL
############################################

def wavelet_stats_singlechannel(j, folder_path):
    """
    Returns mean, std as (4,) CPU float tensors for single-channel wavelets at scale j.
    'folder_path' is used to guess which wavelet stats to load.

    You can either:
      A) Hardcode your own single-channel wavelet stats for each scale j,
         or
      B) Load them from a .json or .npy that you computed offline.

    Below is an *example* showing how you might hardcode the values
    from your "haar" or "db2" results. 
    """
    # For illustration, let's suppose you have stored the stats from your post:
    # (Here we show "haar" with periodization, scale j=1..4).
    # We'll do a minimal example. Adjust as needed:
    if "haar" in folder_path:
        # Hardcode from your posted results:
        # For example, scale 1 means (cH, cV, cD, cA)
        # means = [-6.44258758e-01, -4.80900146e-03, -1.01109268e-04,  2.20622858e+02]
        # stds  = [12.09018123,       2.00253621,       1.55342549,     125.15777866]
        # ...
        if j == 1:
            mean_vals = [-6.44258758e-01, -4.80900146e-03, -1.01109268e-04, 2.20622858e+02]
            std_vals  = [12.09018123,     2.00253621,      1.55342549,     125.15777866]
        elif j == 2:
            mean_vals = [-2.56280647e+00, -1.91221586e-02, -1.34199703e-04, 4.41245716e+02]
            std_vals  = [30.4341562,      5.31983098,      4.38361536,      248.34962495]
        elif j == 3:
            mean_vals = [-1.17511305e+01, -8.21256773e-02, 5.18114617e-03, 8.82491431e+02]
            std_vals  = [72.74251875,     12.73150214,     12.05009051,    490.89030359]
        elif j == 4:
            mean_vals = [-3.64577522e+01, -3.04577641e-01, -2.47833361e-02, 1.76498285e+03]
            std_vals  = [158.73086403,    26.01142649,     28.27669487,     967.41565075]
        else:
            raise ValueError(f"No singlechannel stats available for j={j} with haar.")
    elif "db2" in folder_path:
        # Fill in similarly from your post
        if j == 1:
            mean_vals = [6.44263767e-01, 4.81371286e-03, -1.01098628e-04, 2.20622859e+02]
            std_vals  = [14.72321702,    2.01189994,     1.12941713,      124.87982526]
        elif j == 2:
            mean_vals = [1.27380315e+00, 9.54012155e-03, 2.23646034e-04, 4.41245720e+02]
            std_vals  = [39.07012249,    5.32646268,     3.64512491,      246.59710814]
        elif j == 3:
            mean_vals = [-1.69867007e+00, 8.09610929e-04, 5.20689354e-03, 8.82491446e+02]
            std_vals  = [95.65936512,     12.97643104,    11.14474396,    483.52292248]
        elif j == 4:
            mean_vals = [-9.85226038e+00, -9.10021355e-02, -6.07726359e-03, 1.76498290e+03]
            std_vals  = [219.47397586,    27.74074675,     28.72744086,    940.91338078]
        else:
            raise ValueError(f"No singlechannel stats for j={j} db2.")
    else:
        # Fallback if you have other wavelets or border modes:
        raise ValueError(f"Could not infer single-channel wavelet stats from folder '{folder_path}'.")
    
    return torch.tensor(mean_vals, dtype=torch.float), torch.tensor(std_vals, dtype=torch.float)


############################################
# WaveletDataset for SINGLE CHANNEL images
############################################

class WaveletDataset(Dataset):
    """
    A torch Dataset that loads single-channel wavelet .npz files. Each file has:
       key "j{k}" => shape (4,H',W'), subbands: [cH, cV, cD, cA]
    For the scale self.j, we load "j{self.j}" from each .npz.

    If conditional=True, we learn hi-freq (3 subbands) from low-freq (1 subband).
    Specifically:
       target => shape (3, H', W') = [cH,cV,cD]
       'conditional' => shape (1, H', W') => [cA]

    If conditional=False, we only keep cA as the target.

    The dataset also normalizes wavelet coefficients by subtracting self.mean
    and dividing by self.std for each subband. 
    """

    def __init__(
        self, 
        wav_paths, 
        j, 
        conditional, 
        shard=0, 
        num_shards=1, 
        debug=False
    ):
        super().__init__()
        # Shard the dataset across multiple MPI ranks:
        self.local_wav = wav_paths[shard::num_shards]
        self.j = j
        self.conditional = conditional
        self.debug = debug

        # We only need one example path to figure out which wavelet stats to load
        # (assuming the entire dataset uses the same wavelet type).
        sample_path = self.local_wav[0] if len(self.local_wav) > 0 else None
        if sample_path is None:
            raise ValueError("No .npz files found for WaveletDataset.")
        
        # Compute or load single-channel wavelet stats (4 sub-bands)
        # e.g. wavelet_stats_singlechannel(...) from above
        folder_name = os.path.dirname(sample_path)
        self.mean, self.std = wavelet_stats_singlechannel(self.j, folder_name)

        if debug:
            print(f"[DEBUG:WaveletDataset] scale j={self.j}, conditional={self.conditional}")
            print(f"[DEBUG:WaveletDataset] example path={sample_path}")
            print(f"[DEBUG:WaveletDataset] Loaded wavelet stats mean={self.mean}, std={self.std}")
            print(f"[DEBUG:WaveletDataset] # local wav files: {len(self.local_wav)}")

    def __len__(self):
        return len(self.local_wav)

    def __getitem__(self, idx):
        path = self.local_wav[idx]
        # The .npz has e.g. "j1" -> shape(4,H',W'), "j2"-> etc.
        npz_dict = np.load(path)
        key = f"j{self.j}"
        if key not in npz_dict:
            raise ValueError(f"Missing key '{key}' in npz file: {path}")
        wave_coeffs = npz_dict[key]  # shape (4,H',W')
        wave_coeffs = torch.from_numpy(wave_coeffs).float()  # => (4, H', W')

        # Normalize
        # wave_coeffs[k] = (wave_coeffs[k] - mean[k]) / std[k], for k in [0..3].
        # So we do shape: wave_coeffs => (4,H',W'), mean => (4,), broadcasting
        wave_coeffs -= self.mean[:, None, None]
        wave_coeffs /= self.std[:, None, None]

        # Single-channel => wave_coeffs shape = (4,H',W') => [cH,cV,cD,cA]
        cH = wave_coeffs[0]
        cV = wave_coeffs[1]
        cD = wave_coeffs[2]
        cA = wave_coeffs[3]

        # Return target & dict
        if self.conditional:
            # target = (3,H',W') => hi freq
            # cond = (1,H',W') => cA
            target = torch.stack([cH, cV, cD], dim=0)
            out_dict = {"conditional": cA.unsqueeze(0)}
        else:
            # target = cA => shape (1,H',W')
            target = cA.unsqueeze(0)
            out_dict = {}

        return target, out_dict


############################################
# Generating wavelet coefficients
############################################

def generate_wav_dataset(image_dir, wav_dir, J, wavelet, border_condition):
    """
    Computes the wavelet coefficients of the images in image_dir and dumps them into wav_dir.
    Each output .npz has keys: j1 -> (4, H1, W1), j2 -> (4,H2,W2), ... up to jJ.

    SINGLE-CHANNEL version: 
      - We open each image as grayscale (1,H,W).
      - For each scale, we do dwt2 => (cA,(cH,cV,cD)) => then stack them in a shape (4,H',W') in order [cH,cV,cD,cA].
    """
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)

    # We'll assume you stored grayscale images or something. 
    filelist = sorted(os.listdir(image_dir))
    bar = tqdm.tqdm(filelist, desc="Generating wavelet .npz")
    for filename in bar:
        # Only process images with extension .jpg, .png, etc.
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in ['.jpg', '.png']:
            continue
        img_path = os.path.join(image_dir, filename)

        # Load single-channel
        with Image.open(img_path).convert("L") as im:  # => grayscale
            arr = np.array(im, dtype=np.float32)  # shape (H,W), [0..255 or 0..1, etc.]

        # The following "image_to_wavelets" does J-level decomposition
        # storing (4,H',W') for each scale. 
        wave_coeffs_scales = image_to_wavelets_singlechannel(
            arr, J, wavelet, border_condition
        )

        # Save them to .npz
        out_path = os.path.join(wav_dir, os.path.splitext(filename)[0] + ".npz")
        out_dict = {}
        for sc in range(J):
            key = f"j{sc+1}"
            out_dict[key] = wave_coeffs_scales[sc]
        np.savez(out_path, **out_dict)


def image_to_wavelets_singlechannel(img_array, J, wavelet, border_condition):
    """
    SINGLE-CHANNEL wavelet decomposition for J levels.
    Each level we do: 
       cA, (cH,cV,cD) = dwt2(...)
       we store subbands as shape (4, H', W') => [cH,cV,cD,cA]
       then cA becomes the input for the next scale.

    Return a list of length J, each is a numpy array with shape (4,H',W').
    """
    # If shape is (H,W), that's fine for dwt2. 
    # We'll do it iteratively.
    out_scales = []
    current = img_array  # shape(H,W)
    for _ in range(J):
        cA, (cH, cV, cD) = dwt2(current, wavelet=wavelet, mode=border_condition)
        # stack => shape (4,H',W')
        subbands = np.stack([cH, cV, cD, cA], axis=0)
        out_scales.append(subbands)
        # update current => cA
        current = cA
    return out_scales


############################################
# Reconstructing an image from wavelet coeffs
############################################

def wavelet_to_image_singlechannel(wave_coeffs, border_condition, wavelet, output_size=None):
    """
    Reverses a single scale dwt2, if wave_coeffs has shape (4,H',W').
    i.e. wave_coeffs = [cH,cV,cD,cA], then do:
         idwt2((cA, (cH,cV,cD)), wavelet=..., mode=...)

    If you had done multiple scales, you'd have to run cA up the chain. 
    This function is just for single-scale demonstration.

    :param wave_coeffs: shape (4,H',W')
    :param output_size: if you want to slice the output to a certain size
    :return: (H,W) array
    """
    cH = wave_coeffs[0]
    cV = wave_coeffs[1]
    cD = wave_coeffs[2]
    cA = wave_coeffs[3]
    ret = idwt2((cA, (cH,cV,cD)), wavelet=wavelet, mode=border_condition)
    if output_size is not None:
        ret = ret[:output_size, :output_size]
    return ret


############################################
# Debugging / main
############################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", help="Directory of original single-channel images")
    parser.add_argument("--wav-dir", help="Output directory for wavelet .npz files")
    parser.add_argument("--J", type=int, default=4, help="Max wavelet decomposition scale")
    parser.add_argument("--wavelet", default="haar", help="Wavelet name (e.g. 'haar','db2',...)")
    parser.add_argument("--border-condition", default="periodization", help="mode for pywt dwt2 (periodization,symmetric,...)")
    args = parser.parse_args()

    # Example usage: python wavelet_datasets.py --image-dir=... --wav-dir=... --J=4 --wavelet=haar --border-condition=periodization
    print(f"Generating single-channel wavelet dataset from {args.image_dir} => {args.wav_dir}, J={args.J}, wavelet={args.wavelet}, mode={args.border_condition}")
    generate_wav_dataset(args.image_dir, args.wav_dir, args.J, args.wavelet, args.border_condition)
    print("Done.")

