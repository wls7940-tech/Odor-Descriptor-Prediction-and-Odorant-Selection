import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset

import pandas as pd
import numpy as np

class Spec2Emb(nn.Module):
    def __init__(self, num_emb:int=1000, emb_dim:int=500):
        super(Spec2Emb, self).__init__()
        self.max_exp = 6
        self.emb_con = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.emb_cen = nn.Embedding(
            num_embeddings=num_emb,
            embedding_dim=emb_dim,
        )
        self.trip_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # def _compute_embedding(self, mzs, intens, masks, power):
    #     embs = self.emb_cen(mzs)
    #     embs = embs * masks.unsqueeze(-1)
    #     intens = pt.pow(intens, power).unsqueeze(-1)
    #     embs = (embs * intens).sum(dim=1)
    #     return embs

    def _compute_embedding(self, mzs, intens, masks, power):
        device = self.emb_cen.weight.device
        
        if masks.device != device:
            masks = masks.to(device)
            
        if mzs.device != device:
            mzs = mzs.to(device)


        embs = self.emb_cen(mzs) 
        
        if intens.device != device:
            intens = intens.to(device)

        embs = embs * masks.unsqueeze(-1)
        intens = torch.pow(intens, power).unsqueeze(-1)
        embs = (embs * intens)
        return embs

    def forward(self, data, mode:str='train', power:float=0.5):
        if mode == 'train': 
            mzs_con, masks_con, poss_cen, batch_idx, negs_cen, masks_neg = data
            embs_con = self.emb_con(mzs_con)        # [batch, seq, emb_dim]
            embs_pos = self.emb_cen(poss_cen)     # [B, emb_dim]
            embs_neg = self.emb_cen(negs_cen)      # [B, neg_num, emb_dim]
            embs_neg *= masks_neg.unsqueeze(-1)
            # for every cen word its context words
            embs_con = embs_con[batch_idx] * masks_con.unsqueeze(-1)
            embs_con = embs_con.sum(dim=1) / masks_con.sum(dim=1).unsqueeze(-1) # [B, emb_dim]
            pos_score = (embs_con * embs_pos).sum(dim=-1) # 点积
            pos_score = torch.clamp(pos_score, max=self.max_exp, min=-self.max_exp)
            pos_score = -F.logsigmoid(pos_score)
            neg_score = torch.bmm(embs_neg, embs_con.unsqueeze(-1)).squeeze(-1) # 
            neg_score = torch.clamp(neg_score, max=self.max_exp, min=-self.max_exp)
            neg_score = -F.logsigmoid(-neg_score).sum(dim=-1)
            return (pos_score + neg_score).sum() 
        elif mode == 'emb': # emb模式下的masks只mask掉了padding 
            mzs_all, intens_all, masks_all = data  # [batch, seq]
            return self._compute_embedding(mzs_all, intens_all, masks_all, power)
        elif mode == 'finetune':
            data_mea, data_pre_hit, data_pre_nhit = data
            embs_mea = self._compute_embedding(*data_mea, power)
            embs_pre_hit = self._compute_embedding(*data_pre_hit, power)
            embs_pre_nhit = self._compute_embedding(*data_pre_nhit, power)
            # batchsize, emb_dim
            embs_mea = F.normalize(embs_mea, p=2, dim=-1)
            embs_pre_hit = F.normalize(embs_pre_hit, p=2, dim=-1)
            embs_pre_nhit = F.normalize(embs_pre_nhit, p=2, dim=-1)
            # batchsize
            loss = self.trip_loss(embs_mea, embs_pre_hit, embs_pre_nhit)
            return loss
        else:
            raise ValueError('mode not exist')

class MSTransformer(nn.Module):
    def __init__(self, base_encoder, d_model=500, nhead=4, num_layers=2):
        super().__init__()
        self.base_encoder = base_encoder 
        
        # 2. [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            norm_first=True, 
            # dim_feedforward=d_model*4,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, ms_inputs, mode='emb', power = 0.5):
       
        mzs, intens, raw_masks = ms_inputs
        batch_size = mzs.size(0)
        device = mzs.device


        x = self.base_encoder(ms_inputs, mode='emb', power = 0.5) 

      
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # mw_tokens = self.mw_encoder(mw_inputs).unsqueeze(1) # (B, 1, 500)
        # x = torch.cat((cls_tokens, mw_tokens, x), dim=1) # (B, Seq+2, 500)
        x = torch.cat((cls_tokens, x), dim=1) # (B, Seq+2, 500)

        
        padding_mask = (raw_masks == 0) # (B, Seq)
    
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    
        extended_mask = torch.cat((cls_mask, padding_mask), dim=1) # (B, Seq+2)

        x_out = self.transformer(x, src_key_padding_mask=extended_mask)

        final_emb = x_out[:, 0, :]
        
        return self.out_norm(final_emb)
