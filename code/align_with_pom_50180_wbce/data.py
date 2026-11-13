import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class ClassificationData(Dataset):
    def __init__(self, ms_df, label=None, chem_emb=None):
        
        if label is not None and 'nonStereoSMILES' in label.columns:
            self.smiles = label['nonStereoSMILES'].tolist()
        elif 'canonical_smiles' in ms_df.columns:
            self.smiles = ms_df['canonical_smiles'].tolist()
        else:
            self.smiles = [f"mol_{i}" for i in range(len(ms_df))]

        self.ms_raw = ms_df.iloc[:, 6 : 1006].values 

        self.has_label = label is not None
        if self.has_label:
            self.label = label.iloc[:, 3:].values
            
        self.has_chem = chem_emb is not None
        if self.has_chem:
            if isinstance(chem_emb, pd.DataFrame):
                self.chem_emb = chem_emb.values.astype(np.float32)
            else:
                self.chem_emb = chem_emb.astype(np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]

        ms_raw_vector = self.ms_raw[index] 
        ms_mol = to_model_data(ms_raw_vector.reshape(1, -1))[0]
        out_label = self.label[index, :] if self.has_label else np.array([-1.0], dtype=np.float32)
        out_chem = self.chem_emb[index, :] if self.has_chem else np.array([-1.0], dtype=np.float32)

        return smiles, ms_mol, out_label, out_chem
    
def classification_collate(batch, augmentor = None):
    smiles_list = []
    ms_raw_list = []
    label_list = []
    chem_list = []
    processed_specs = []
    
    for smiles, ms_mol, label, chem_emb in batch:
        smiles_list.append(smiles)
        ms_raw_list.append(ms_mol)
        label_list.append(label)
        if isinstance(chem_emb, np.ndarray):
            chem_list.append(torch.from_numpy(chem_emb).float()) # 转换为 Tensor
        else:
            chem_list.append(chem_emb.float())

        mz = ms_mol.mz
        inten = ms_mol.intensities
        if augmentor:
            mz, inten = augmentor.augment(mz, inten)
        
        processed_specs.append((mz, inten))
    
    # max_len = max([len(x.mz) for x in ms_raw_list])
    if len(processed_specs) > 0:
        max_len = max([len(spec[0]) for spec in processed_specs])
    else:
        max_len = 0
    
    mzs_padded = []
    intens_padded = []
    masks_padded = []
    # for mol_spec in ms_raw_list:
    #     mz = mol_spec.mz
    #     inten = mol_spec.intensities 

        
    #     len_mz = len(mz)
    #     pad_num = max_len - len_mz
        
    #     mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
    #     inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
    #     mask = np.pad(np.ones_like(mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

    #     mzs_padded.append(mz_con)
    #     intens_padded.append(inten_con)
    #     masks_padded.append(mask)

    # mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    # intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    # masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    # ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
    # labels_np = np.stack(label_list)
    # # labels_indices_np = np.argmax(labels_np, axis=1) 
    # labels_tensor = torch.tensor(labels_np, dtype=torch.float)

    # chem_tensor = torch.stack(chem_list)

    # return smiles_list, ms_inputs, labels_tensor, chem_tensor
    for mz, inten in processed_specs:
        len_mz = len(mz)
        pad_num = max_len - len_mz
        
        if pad_num < 0:
            pad_num = 0
        

        mz_con = np.pad(mz, (0, pad_num), mode='constant', constant_values=0)
        inten_con = np.pad(inten, (0, pad_num), mode='constant', constant_values=0)
        
        mask = np.pad(np.ones(len_mz, dtype=bool), (0, pad_num), mode='constant', constant_values=False)

        mzs_padded.append(mz_con)
        intens_padded.append(inten_con)
        masks_padded.append(mask)

    mzs_tensor = torch.tensor(np.array(mzs_padded), dtype=torch.long)
    intens_tensor = torch.tensor(np.array(intens_padded), dtype=torch.float)
    masks_tensor = torch.tensor(np.array(masks_padded), dtype=torch.bool)

    ms_inputs = (mzs_tensor, intens_tensor, masks_tensor)
    
    labels_np = np.stack(label_list)
    labels_tensor = torch.tensor(labels_np, dtype=torch.float)

    chem_tensor = torch.stack(chem_list)

    return smiles_list, ms_inputs, labels_tensor, chem_tensor

class Classification(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 138, hidden_dim_ratio: float = 0.5, dropout_rate: float = 0.5):
        super(Classification, self).__init__()
        
        hidden_dim = int(input_dim * hidden_dim_ratio)
        
        if hidden_dim < 1:
            hidden_dim = 1
            
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.fc2(x)
        
        return logits