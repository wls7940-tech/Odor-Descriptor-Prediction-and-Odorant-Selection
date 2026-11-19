Odor Descriptor Prediction and Odorant Selection
========================================================

This is a guidance to use downstream classification head of SCENT. 
2026.05.20, by Ziqi Zhang from KTH Royal Institute of Technology,
2026.11.13 revised by Eunyeong Jin from Max Planck Institute for Chemistry

**Environment setup**
You may need to use terminal tool in your PC to install the environment.

1. Create a clean environment
```bash
conda create -n scent python=3.10
conda activate scent
```

2. Install the required packages
```bash
pip install -r requirements.txt
```

3. PyTorch installation
```bash
pip install torch==2.12.0
```
The code was tested with PyTorch 2.12.0 (CPU version). For GPU users, CUDA-specific installation instructions are available at https://pytorch.org/get-started/locally/

You could test whether PyTorch are available in terminal.
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```
-------------------
## Run the code (prediction_selection.py)

### 1. Prepare the input files

Place the required files in the following repository structure:

```text
Odor-Descriptor-Prediction-and-Odorant-Selection/
├── README.md
├── requirements.txt
├── GT_vectors.xlsx
├── data/
│   └── limonene.csv
├── checkpoints/
│   └── clas_fold1_frac_pom_50180_wbce_all.pt
└── code/
    ├── prediction_selection.py
    ├── data.py
    └── utils_ms.py
```

- `data/`: Raw mass spectrometry CSV files exported from the instrument.
- `checkpoints/`: Pretrained model checkpoint files.
- `GT_vectors.xlsx`: Ground-truth odor descriptor vectors.

The Excel file must contain the following columns:
`odorant`, `descriptor`, and `weight`.

All file paths are defined relative to the repository root. No manual path modification is required.

### 2. Open a terminal in the repository root directory

Make sure the `scent` environment is activated.

If necessary, activate it using:

```bash
conda activate scent
```

### 3. Run the prediction script

```bash
python code/prediction_selection.py
```

When prompted, enter the CSV filename (e.g., `limonene.csv`).

The script automatically reads the CSV file from the `data/` directory.

### 4. Interpret the results

The script predicts odor descriptors from the input mass spectrum and compares the resulting prediction vector with the GT vectors using angular distance.

Angular distance is calculated in radians, with smaller values indicating greater similarity.

The odorant with the smallest angular distance is displayed as the best match.

-------------------
## Reference
[1] Zhang, Z. et al. SCENT: Aligning Mass Spectra with Molecular Structure for Olfactory Perception. Preprint at https://doi.org/10.48550/ARXIV.2605.27009 (2026)

[2] Good Scents Company. The Good Scents Company Information System. https://www.thegoodscentscompany.com/
