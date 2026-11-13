Odor Descriptor Prediction and Odorant Selection
========================================================

This is a guidance to use downstream classification head of SCENT. 
2026.05.20, by Ziqi Zhang from KTH Royal Institute of Technology,
2026.11.13 revised by Eunyeong Jin from Max Planck Institute for Chemistry

**Environment setup**
You may need to use terminal tool in your PC to install the environment.

1. Create a clean environment
```bash
conda activate -n scent python=3.10
conda activate scent
```

2. Install the required packages
```bash
pip intsall -r requirements.txt
```

3. PyTorch installation
```bash
pip install torch
```
You could test whether PyTorch are available in terminal.
```bash
import torch
print('PyTorch version:', torch.__version__)
```
-------------------
**Run the code (prediction_selection.py)**
1. Change the code to your own PC path
Line 30: this is the path where you put the model files.
```python
base_path = '.../code/align_with_pom_50180_wbce/'
```
Line 793: this is the path where you put the data files. You should use the RAW data file (.CSV) for this code (the data derive from instrument)
```python
base_dir = '...'
```

2. Check GT (Ground-truth) vector file: Place GT_vectors.xlsx in the same directory as the Python script.

The Excel file must contain these columns:
```python
    odorant
    descriptor
    weight
``` 
Line 497: Replace it with the directory containing the GT file.
```python
r"path/to/your/directory/GT_vectors.xlsx"
``` 

3. cd your terminal to your code path
```bash
cd ...the root path of where prediction_selection.py locates
```
Please make sure your environment is still activate. If not, please run
```bash
conda activate scent
```

4. Run the code in the terminal
```bash
python prediction_selection.py
```

5. Angular distance is calculated in radians: Smaller values indicate greater similarity. The name of the compound whose GT vector is closest to the predicted vector is displayed.
```python
    angular_distance = arccos(cosine_similarity)
```

-------------------
## Reference
[1] Zhang, Z. et al. SCENT: Aligning Mass Spectra with Molecular Structure for Olfactory Perception. Preprint at https://doi.org/10.48550/ARXIV.2605.27009 (2026)

[2] Good Scents Company. The Good Scents Company Information System. https://www.thegoodscentscompany.com/
