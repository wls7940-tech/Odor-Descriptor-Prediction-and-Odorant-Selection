import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os
import argparse
import pandas as pd
import numpy as np
import torch
import math

from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data import ClassificationData, classification_collate, Classification
from utils_ms import Spec2Emb, MSTransformer

from IPython.core.display_functions import display
from matplotlib.patches import Patch
import numpy as np
from sklearn import manifold
from tqdm import tqdm
from pathlib import Path

# ============================================================
# REPOSITORY PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
GT_FILE = ROOT / "GT_vectors.xlsx"

device = 'cpu'
seed = 42


base_path = CKPT_DIR
ckpt_template = 'clas_fold1_frac_pom_50180_wbce_all.pt'


INPUT_DIM = 500
OUTPUT_DIM = 138
NUM_FOLDS = 1


# labels = pd.read_csv(label_path)
label_names = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
    ]

def signal_processing(file_path):
    # read csv file
    file_base_name = os.path.splitext(os.path.basename(file_path))[0]
    data_rows = []
    with open(file_path, 'r') as f:
        for line in f:
            if any(keyword in line for keyword in ['[data', '[format', '[spectra]']):
                continue
            parts = line.strip().split(',')
            data_rows.append(parts)

    raw_ms = pd.DataFrame(data_rows)
    
    # extract time, mz and intensity
    times = raw_ms.iloc[7:, 0]
    mz_values = raw_ms.iloc[7:, 7::2]
    int_values = raw_ms.iloc[7:, 8::2]

    times = pd.to_numeric(times, errors='coerce').values

    mz_values = mz_values.apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    int_values = int_values.apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    mz_values = np.nan_to_num(mz_values.astype(float), nan=0.0)
    int_values = np.nan_to_num(int_values.astype(float), nan=0.0)

    # automatically detect signal
    tic = int_values.sum(axis=1)
    results = detect_by_slope_refined(times, tic)

    # plt.pause(0.1)

    intensity_matrix, mz_axis = to_intensity_matrix(mz_values, times, int_values)

    ms_clean_df = denoise_round(intensity_matrix, results, mz_axis, times)
    mz_round = ms_clean_df.columns.astype(float).values

    ms_time_average = sample_ave_signals(results, ms_clean_df, times, file_base_name)

    # plt.pause(0.1)
    # ms_time_average = (ms_time_average * 999) / max(ms_time_average)

    spe_min = ms_time_average.min()
    spe_max = ms_time_average.max()
    

    spe_range = spe_max - spe_min
    
    if spe_range == 0:
        spe_range = 1

    normalized_spe = ((ms_time_average - spe_min) / spe_range) * 999
    normalized_spe = normalized_spe.round().astype(int)

    return normalized_spe


def denoise_round(intensity_matrix, results, mz_axis, times):
    bg_part1 = intensity_matrix[:results['start_idx'], :]
    bg_part2 = intensity_matrix[max(results['end_idx'] if results['end_idx'] is not None else 0, results['tail_idx'] if results['tail_idx'] is not None else 0):, :]

    bg_combined = np.vstack((bg_part1, bg_part2))

    bg_nan = np.where(bg_combined == 0, np.nan, bg_combined)

    baseline = np.nan_to_num(np.nanmean(bg_nan, axis=0))

    std = np.nan_to_num(np.nanstd(bg_nan, axis=0))

    n_sigma = 5 
    threshold = baseline + (n_sigma * std)
    int_denoise = intensity_matrix - threshold
    int_denoise = np.maximum(int_denoise, 0)
    mz_rounded_headers = [clean_and_round(val) for val in mz_axis]

    df_intensity_matrix = pd.DataFrame(int_denoise, index=times, columns=mz_rounded_headers)

    df_intensity_matrix = df_intensity_matrix.loc[:, df_intensity_matrix.columns.notna()]

    df_intensity_matrix = df_intensity_matrix.groupby(level=0, axis=1).sum()

    return df_intensity_matrix

def to_intensity_matrix(mz_values, times, int_values):
    min_mz = np.floor(mz_values[mz_values > 0].min().min())
    max_mz = np.ceil(mz_values.max().max())

    mz_axis = np.around(np.arange(min_mz, max_mz+ 0.1, 0.1), decimals=1)
    mz_to_idx = {round(val, 1): i for i, val in enumerate(mz_axis)}

    intensity_matrix = np.zeros((len(times), len(mz_axis)))

    for i in range(mz_values.shape[0]):
        row_mz = mz_values[i]
        row_int = int_values[i]

        for m, intensity in zip(row_mz, row_int):
            if m > 0:
                rounded_m = round(m, 1)
                if rounded_m in mz_to_idx:
                    idx = mz_to_idx[rounded_m]
                    intensity_matrix[i, idx] += intensity
    return intensity_matrix, mz_axis

def detect_by_slope_refined(times, intensity, window_size=15, n_sigma=4): 
    intensity = np.asarray(intensity)
    times = np.asarray(times)
    slopes = np.zeros(len(intensity))

    for i in range(window_size, len(intensity)):
        y = intensity[i-window_size : i]
        x = times[i-window_size : i]
        slope, _ = np.polyfit(x, y, 1)
        slopes[i] = slope

    valid_slopes = slopes[window_size:] 
    safe_slopes = valid_slopes[abs(valid_slopes) < np.percentile(abs(valid_slopes), 50)]
    start_threshold = n_sigma * np.std(safe_slopes) 
    
    diffs = np.abs(np.diff(intensity))
    global_noise_std = np.median(diffs) / 0.6745
    if global_noise_std == 0: 
        global_noise_std = np.std(intensity[:50]) 
        
    start_idx = None
    peak_base_bg = 0
    peak_base_std = 0
    
    for i in range(window_size, len(slopes)):
        
        lookback_start = max(0, i - 100)
        lookback_end = max(1, i - 5)
        recent_data = intensity[lookback_start : lookback_end]
        
        if len(recent_data) < 5:
            local_bg_mu = np.mean(intensity[:5])
        else:
            local_bg_mu = np.percentile(recent_data, 30)
            
        intensity_threshold = local_bg_mu + 5 * global_noise_std

        if slopes[i] > max(start_threshold, 0) and intensity[i] > intensity_threshold:
            start_idx = max(0, i - math.ceil(window_size / 2))
            

            peak_base_bg = local_bg_mu
            peak_base_std = global_noise_std 
            break

    steep_drop_idx = None
    end_idx = None
    tail_idx = None 
    
    if start_idx is not None:
        peak_idx = np.argmax(intensity[start_idx:]) + start_idx
        steep_drop_idx = np.argmin(slopes[peak_idx:]) + peak_idx
        
        peak_max_height = intensity[peak_idx] - peak_base_bg
        end_idx = len(intensity) - 1  
        
        for j in range(steep_drop_idx, len(intensity)):
            current_net_height = intensity[j] - peak_base_bg
            
            if current_net_height < 3 * peak_base_std or \
               (current_net_height < 0.05 * peak_max_height and abs(slopes[j]) < np.std(safe_slopes)):
                end_idx = j
                break
  
        if (end_idx - steep_drop_idx) < 10:
            tail_idx = end_idx
        else:
            accel = np.gradient(slopes)
            tail_candidates = np.where((accel > 0) & 
                                       (np.arange(len(accel)) > steep_drop_idx) & 
                                       (np.arange(len(accel)) < end_idx))[0]
            
            if len(tail_candidates) > 0:
                candidate_idx = tail_candidates[0]
                current_height = intensity[candidate_idx] - peak_base_bg
                if current_height < (0.05 * peak_max_height) or intensity[candidate_idx] < (peak_base_bg + 2 * peak_base_std):
                    tail_idx = end_idx
                else:
                    tail_idx = candidate_idx
            else:
                tail_idx = end_idx

    return {
        'start_idx': start_idx,
        'steep_drop_idx': steep_drop_idx,
        'tail_idx': tail_idx,
        'end_idx': end_idx
    }

def plot_tic(tic, times, results):
    plt.figure(figsize=(12, 6))
    plt.plot(times, tic, label='TIC')

    if results['start_idx']:
        plt.vlines(times[results['start_idx']], 0, max(tic), 'r', '--', label='Start')
    if results['steep_drop_idx']:
        plt.vlines(times[results['steep_drop_idx']], 0, max(tic), 'g', ':', label='Steepest Drop')
    if results['tail_idx']:
        plt.vlines(times[results['tail_idx']], 0, max(tic), 'orange', '--', label='Tailing Start')
    if results['end_idx']:
        plt.vlines(times[results['end_idx']], 0, max(tic), 'purple', '--', label='Signal End')

    plt.legend()
    plt.show()

def clean_and_round(x):
    try:
        return int(np.round(float(str(x).replace('#', '').strip())))
    except:
        return np.nan

def sample_ave_signals(results, intensity_matrix, times, file_base_name):
    start_idx = results['start_idx']
    

    stop_candidates = [x for x in [results['tail_idx'], results['steep_drop_idx']] if x is not None]
    if stop_candidates:
        stop_idx = min(stop_candidates)
    else:
        stop_idx = len(times) 


    if start_idx is not None and start_idx < stop_idx:

        ave_spe = intensity_matrix.iloc[start_idx : stop_idx, :].mean(axis=0)
    else:
        print(f"  Warning: Signal detection failed or range invalid for {file_base_name}. Using full range average.")
        ave_spe = intensity_matrix.mean(axis=0)
    
    return ave_spe.values

def plot_eic_mean(mz_axis, ave_spe):
    plt.figure(figsize=(12, 5))
    plt.stem(mz_axis, ave_spe, linefmt='b-', markerfmt=' ', basefmt=' ')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title('Average signal')
    plt.grid(True, alpha=0.3)
    plt.show()

def load_fold_model(fold, base_path=base_path, device=device):
    """Load one fold using the same structure as the validation ensemble code."""
    base_encoder = Spec2Emb(num_emb=1000, emb_dim=500)
    model_fold = MSTransformer(
        base_encoder=base_encoder,
        d_model=500,
        nhead=4,
        num_layers=2
    ).to(device)

    ckpt_path = os.path.join(base_path, ckpt_template.format(fold=fold))
    ckpt = torch.load(ckpt_path, map_location=device)

    model_fold.load_state_dict(ckpt['ms_encoder'])
    model_fold.eval()

    head = Classification(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
    head.load_state_dict(ckpt['classification'], strict=False)
    head.eval()

    return model_fold, head

def extract_vectors_external_trans(model, classification_head, dataloader, device):

    model.eval()
    classification_head.eval()
    ms_embeddings = []
    all_logits = []
    all_sample_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting (Transformer)"):
            smiles_batch = batch[0]
            ms_batch = batch[1]

            mzs, intens, masks = ms_batch
            ms_inputs = (
                mzs.to(device, non_blocking=True),
                intens.to(device, non_blocking=True),
                masks.to(device, non_blocking=True)
            )

            ms_emb = model(ms_inputs, mode='emb')
            logits = classification_head(ms_emb)

            all_logits.append(logits.cpu())
            ms_embeddings.append(ms_emb.cpu().numpy())
            all_sample_names.extend(list(smiles_batch))

    all_logits_tensor = torch.cat(all_logits)
    all_probs_np = torch.sigmoid(all_logits_tensor).numpy()
    all_embeddings_np = np.concatenate(ms_embeddings, axis=0)

    return all_embeddings_np, all_probs_np, all_sample_names

def print_prediction_results(all_probs_np, label_names, sample_names=None, threshold=0.3, fallback_top_k=5, max_samples=41):
    """Print odor label predictions from probability outputs."""
    for i in range(min(max_samples, len(all_probs_np))):
        probs = all_probs_np[i]

        pred_indices = np.where(probs > threshold)[0]
        has_high_confidence = len(pred_indices) > 0

        if has_high_confidence:
            sorted_order = np.argsort(probs[pred_indices])[::-1]
            pred_indices = pred_indices[sorted_order]
        else:
            pred_indices = np.argsort(probs)[::-1][:fallback_top_k]

        pred_words = [f"{label_names[idx]} ({probs[idx]:.2f})" for idx in pred_indices]
        sample_name = sample_names[i] if sample_names is not None and i < len(sample_names) else i

        print(f"sample {sample_name}:")
        if has_high_confidence:
            print(f"  👉 model prediction: {pred_words}")
        else:
            print(f"  ⚠️ The model has less confidence in the current sample and tends to favor the following labels: {pred_words}")
        print("-" * 40)

def predict_with_ensemble(
    dataloader,
    label_names,
    device,
    base_path=base_path,
    num_folds=NUM_FOLDS,
    threshold=0.3
):

    all_fold_probs = []
    all_fold_embeddings = []

    sample_names = None


    all_pred_words = []
    all_pred_probs = []

    # =========================
    # FOLD LOOP
    # =========================
    for fold in range(1, num_folds + 1):

        model_fold, head = load_fold_model(
            fold,
            base_path=base_path,
            device=device
        )

        emb, probs, current_sample_names = extract_vectors_external_trans(
            model_fold,
            head,
            dataloader,
            device
        )

        all_fold_embeddings.append(emb)
        all_fold_probs.append(probs)

        if sample_names is None:
            sample_names = current_sample_names

        print(f"✅ fold {fold} finished")

    # =========================
    # ENSEMBLE
    # =========================
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    ensemble_emb = np.mean(all_fold_embeddings, axis=0)

    print(f"Ensemble shape: {ensemble_probs.shape}")

    # =========================
    # TOP5
    # =========================
    top_idx = np.argsort(ensemble_probs, axis=1)[:, -5:][:, ::-1]

    for i in range(len(ensemble_probs)):

        idx = top_idx[i]

        pred_words = [
            f"{label_names[j]} ({ensemble_probs[i][j]:.2f})"
            for j in idx
        ]

        all_pred_words.append(pred_words)
        all_pred_probs.append(ensemble_probs[i][idx])

    print_prediction_results(
        ensemble_probs,
        label_names,
        sample_names=sample_names,
        threshold=threshold
    )

    return (
        ensemble_emb,
        ensemble_probs,
        all_fold_probs,
        all_pred_words,
        all_pred_probs
    )

def to_dataloader(ms_time_average):
    vector = np.zeros(1001)

    if len(ms_time_average.T) == 131:
        vector[50:181] = ms_time_average.flatten()
    else:
        print(f"warning: signal's shape {len(ms_time_average)} is not 131, check signal processing!")
    
    final_input_df = pd.DataFrame([vector], columns=range(1001))
    new_columns = ['empty_1', 'empty_2', 'empty_3', 'empty_4', 'empty_5']

    start_loc = 0


    for i, col_name in enumerate(new_columns):

        final_input_df.insert(loc=start_loc + i, column=col_name, value=np.nan)
    dataset = ClassificationData(final_input_df,)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=classification_collate)
    
    return data_loader


# =========================
# LOAD GT
# =========================
df = pd.read_excel(GT_FILE)

df.columns = df.columns.str.strip()


# =========================
# UTIL
# =========================
def split_terms(text):
    return [
        x.strip().lower()
        for x in str(text).split(";")
        if x.strip()
    ]


def split_numbers(text):
    return [
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]


def normalize(v):
    v = np.array(v, dtype=float)

    s = np.sum(v)

    if s == 0:
        return v

    return v / s


# =========================
# COSINE SIMILARITY
# =========================
def cosine(a, b):

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    cos_sim = np.dot(a, b) / denom

    # Prevent floating-point errors
    cos_sim = np.clip(
        cos_sim,
        -1.0,
        1.0
    )

    return cos_sim


# =========================
# ANGULAR DISTANCE
# =========================
def angular_distance(a, b):

    cos_sim = cosine(a, b)

    # np.arccos returns radians
    angle = np.arccos(cos_sim)

    return angle


# =========================
# BUILD VOCAB (GT + PRED)
# =========================
def build_vocab(all_pred_words):

    vocab = set()

    # GT descriptors
    for _, row in df.iterrows():

        vocab.update(
            split_terms(
                row["descriptor"]
            )
        )

    # Prediction descriptors
    for pred in all_pred_words:

        for item in pred:

            label = (
                item
                .split("(")[0]
                .strip()
                .lower()
            )

            vocab.add(label)

    vocab = sorted(
        list(vocab)
    )

    vocab_index = {
        w: i
        for i, w in enumerate(vocab)
    }

    return vocab, vocab_index


# =========================
# GT VECTOR
# =========================
def build_gt_vec(
    descriptor_text,
    weight_text,
    vocab_index,
    vocab
):

    vec = np.zeros(
        len(vocab)
    )

    descriptors = split_terms(
        descriptor_text
    )

    weights = normalize(
        split_numbers(
            weight_text
        )
    )

    for descriptor, weight in zip(
        descriptors,
        weights
    ):

        if descriptor in vocab_index:

            vec[
                vocab_index[descriptor]
            ] = weight

    return vec


# =========================
# PREDICTION VECTOR (TOP 5)
# =========================
def build_pred_vec(
    pred_words,
    vocab_index,
    vocab
):

    vec = np.zeros(
        len(vocab)
    )

    labels = []
    probs = []

    for item in pred_words:

        label = (
            item
            .split("(")[0]
            .strip()
            .lower()
        )

        prob = float(
            item
            .split("(")[1]
            .replace(")", "")
            .strip()
        )

        labels.append(label)
        probs.append(prob)

    probs = np.array(
        probs,
        dtype=float
    )

    # Normalize top-5 probabilities
    if probs.sum() != 0:
        probs = probs / probs.sum()

    for label, prob in zip(
        labels,
        probs
    ):

        if label in vocab_index:

            vec[
                vocab_index[label]
            ] = prob

    return vec


# =========================
# ANGULAR DISTANCE MATCHING
# =========================
def run_angular_distance(
    all_pred_words
):

    vocab, vocab_index = build_vocab(
        all_pred_words
    )

    results = []

    for i in range(
        len(all_pred_words)
    ):

        pred_vec = build_pred_vec(
            all_pred_words[i],
            vocab_index,
            vocab
        )

        # Smaller angular distance = more similar
        best_angle = float("inf")
        best_odorant = None

        # Compare prediction with every GT vector
        for j in range(len(df)):

            gt_vec = build_gt_vec(
                df.iloc[j]["descriptor"],
                df.iloc[j]["weight"],
                vocab_index,
                vocab
            )

            angle = angular_distance(
                pred_vec,
                gt_vec
            )

            if angle < best_angle:

                best_angle = angle
                best_odorant = df.iloc[j]["odorant"]

        print("\nSample:", i)

        print(
            "BEST MATCH:",
            best_odorant
        )

        print(
            "ANGULAR DISTANCE:",
            round(best_angle, 4),
            "radians"
        )

        results.append(
            best_angle
        )

    print(
        "\n========================"
    )

    print(
        "AVG ANGULAR DISTANCE:",
        round(
            np.mean(results),
            4
        ),
        "radians"
    )

    print(
        "========================"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    base_dir = DATA_DIR

    while True:

        # =========================
        # INPUT
        # =========================
        user_input = input(
            "\nEnter CSV filename (q to quit): "
        )

        if user_input.lower() == "q":
            break

        if user_input.strip() == "":

            filename = "limonene.csv"

        else:

            filename = user_input.strip()

            if not filename.lower().endswith(".csv"):
                filename += ".csv"

        file_path = os.path.join(
            base_dir,
            filename
        )

        print(
            f"\nProcessing: {file_path}"
        )

        # =========================
        # INFERENCE
        # =========================
        test_loader = to_dataloader(
            signal_processing(
                file_path
            )
        )

        (
            emb,
            ensemble_probs,
            all_fold_probs,
            all_pred_words,
            all_pred_probs
        ) = predict_with_ensemble(
            test_loader,
            label_names,
            device,
            base_path=base_path,
            num_folds=NUM_FOLDS,
            threshold=0.3
        )

        # =========================
        # ANGULAR DISTANCE EVALUATION
        # =========================
        run_angular_distance(
            all_pred_words
        )
