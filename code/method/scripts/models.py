import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class DS_MVP_Model(nn.Module):
    def __init__(self):
        super(DS_MVP_Model, self).__init__()
        self.seq_express = Seq_Model()
        self.scores_af_express = Scores_AF_Model()
        self.mgw_shape_express = Shape_Model()

        self.gene_express = Gene_Model()
        self.aa_express = AA_Model()


        self.seq_dim = 7*192
        self.scores_af_dim = 128
        self.shape_dim = 64
        self.gene_dim = 512
        self.aa_dim = 64

        self.mlp_dim = self.seq_dim + self.scores_af_dim + self.shape_dim + self.gene_dim + self.aa_dim

        self.classfier = nn.Sequential(
            nn.Linear(self.mlp_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.sigmoid = nn.Sigmoid()



    def forward(self, feat_lst):
        ref_embed, alt_embed, scores, afs, ref_mgw, alt_mgw, gene_embed, aa_ref, aa_alt = feat_lst
        
        seq_feat = self.seq_express(ref_embed, alt_embed)
        gene_feat = self.gene_express(gene_embed)
        other_feat = self.scores_af_express(scores, afs)
        shape_feat = self.mgw_shape_express(ref_mgw, alt_mgw)
        aa_feat = self.aa_express(aa_ref, aa_alt)

        feat = torch.cat([seq_feat, shape_feat, other_feat, gene_feat, aa_feat], dim=-1)
        tmp_out = feat

        out = self.classfier(feat)
        # out = self.sigmoid(out)

        return out, tmp_out




class Seq_Model(nn.Module):
    def __init__(self, embed_vocab=125, hidden_dim=64, lstm_layer=2, heads=4, attn_drop=0.1, attn_layer=2):
        super(Seq_Model, self).__init__()
        self.embed = nn.Embedding(embed_vocab, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layer, batch_first=True, bidirectional=True)

        self.trm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim*2, batch_first=True, nhead=heads, dropout=attn_drop),
                                                  num_layers=attn_layer)


        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(hidden_dim,hidden_dim, 3, stride=1, padding=1),  # [bs 32 72]
            nn.ReLU(),
            nn.Conv1d(hidden_dim,hidden_dim, 3, stride=1, padding=1),  # [bs 32 72]
        )

        self.flat = nn.Flatten()

    def forward(self, ref_embed, alt_embed):
        ref_embed = self.embed(ref_embed.long())   # [bsz, 97, 128]
        alt_embed = self.embed(alt_embed.long())   # [bsz, 97, 128]

        ref_lstm, _ = self.lstm(ref_embed)              # [n, 97, 256]
        alt_lstm, _ = self.lstm(alt_embed)              # [n, 97, 256]

        ref_trm = self.trm(ref_lstm)
        alt_trm = self.trm(alt_lstm)

        ref_cnn = torch.transpose(ref_lstm, 1, 2)
        alt_cnn = torch.transpose(alt_lstm, 1, 2)
        ref_cnn = self.conv(ref_cnn)
        alt_cnn = self.conv(alt_cnn)
        ref_cnn = torch.transpose(ref_cnn, 1, 2)
        alt_cnn = torch.transpose(alt_cnn, 1, 2)

        ref_concat = torch.cat([ref_trm, ref_cnn], dim=-1)
        alt_concat = torch.cat([alt_trm, alt_cnn], dim=-1)

        seq = ref_concat - alt_concat
        mid = seq.shape[1]//2
        seq = seq[:, mid-3: mid+4,:]
        seq_flat = self.flat(seq)
        return seq_flat



class Scores_AF_Model(nn.Module):
    def __init__(self, s_input_dim=8, s_hidden_dim=32, a_input_dim=2, a_hidden_dim=16, out_dim=128):
        super(Scores_AF_Model, self).__init__()
        ####### af=2
        self.embed_scores = nn.Linear(s_input_dim, s_hidden_dim)
        self.embed_af = nn.Linear(a_input_dim, a_hidden_dim)
        self.fc = nn.Linear(s_hidden_dim + a_hidden_dim, out_dim)

    def forward(self, scores, afs):
        scores = self.embed_scores(scores)
        afs = self.embed_af(afs)
        feat = torch.cat([scores, afs], dim=-1)
        feat = self.fc(feat)

        return feat


class Shape_Model(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, conv_dim = 32, out_dim=64):
        super(Shape_Model, self).__init__()
        
        self.embed = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(hidden_dim,conv_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_dim,conv_dim, 3, stride=1, padding=1)
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(9*conv_dim, conv_dim*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(conv_dim*4, out_dim),
        )


    def forward(self, ref, alt):    
        ref = ref.unsqueeze(dim=-1)     # [n,9,1]
        alt = alt.unsqueeze(dim=-1)

        ref = torch.transpose(ref, 1, 2)
        alt = torch.transpose(alt, 1, 2)

        embed_ref = self.embed(ref)
        embed_alt = self.embed(alt)

        embed_ref = torch.transpose(embed_ref, 1, 2)    # [n,9,32]
        embed_alt = torch.transpose(embed_alt, 1, 2)

        # sub
        mgw_feat = embed_ref - embed_alt
        mgw_flat = self.flat(mgw_feat)
        mgw_flat = self.fc(mgw_flat)
        return mgw_flat



class Gene_Model(nn.Module):
    def __init__(self, in_dim=1536, hidden_dim=1024, out_dim=512):
        super(Gene_Model, self).__init__()


        self.gene_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )
        self.gene_conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, 3, stride=1, padding=1),
        )


    def forward(self, gene_embed): 
        ######### fc
        gene_fc_feat = self.gene_fc(gene_embed)
        gene_fc_feat = gene_fc_feat.unsqueeze(2)
        ######## conv
        gene_embed_squee = gene_embed.unsqueeze(2)      # [n,1536,1]

        gene_conv_feat = self.gene_conv(gene_embed_squee)   # [n ,512, 1]

        gene_feat = gene_fc_feat + gene_conv_feat
        gene_feat = gene_feat.squeeze(-1)
        return gene_feat





class AA_Model(nn.Module):
    def __init__(self, embed_vocab=20, embed_dim=128, out_dim=64):
        super(AA_Model, self).__init__()
        self.aa_embed = nn.Embedding(embed_vocab, embed_dim)

        self.aa_ln = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim,out_dim)
        )


    def forward(self, aa_ref, aa_alt):
        aa_ref_embed = self.aa_embed(aa_ref.long())    # [n,1] -> [n,1, 128]
        aa_alt_embed = self.aa_embed(aa_alt.long())
        aa_feat = aa_ref_embed - aa_alt_embed
        aa_feat = aa_feat.squeeze(1)

        aa_feat = self.aa_ln(aa_feat)

        return aa_feat

