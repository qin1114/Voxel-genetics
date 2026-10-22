"""
voxelgenetics.voxel.gencov
--------------------------
Voxel-wise genetic covariance (gencov) and genetic correlation (rg) 
estimation using LDSC and PyTorch (GPU-accelerated).
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch

from ..ldsc.regression import Hsq, Gencov
from ..ldsc.jackknife import RatioJackknife
from .h5_io import load_h5_snplist


def estimate_rg_voxel(
    chromosome_range,
    ld_score_folder,
    target_gwas_path,
    gwas_h5_template=None,
    bwas_dict=None,
    common_snps_list=None,
    m_annot=1173569,
    chunk_size=500,
    n_blocks=200,
    twostep=30,
    device='cuda:0',
    target_snp_col='SNP',
    target_z_col='Z',
    target_n_col='N'
):
    """
    Estimate voxel-wise genetic covariance (gencov) and genetic correlation (rg) 
    between brain voxels and a target phenotype.

    Parameters
    ----------
    chromosome_range : iterable
        List of chromosome numbers (e.g., range(1, 23)).
    ld_score_folder : str
        Directory containing LD score files.
    target_gwas_path : str
        Path to the target phenotype GWAS summary statistics file.
    gwas_h5_template : str, optional
        Path template for HDF5 files (e.g., '..._chr{chr}_train.h5').
    bwas_dict : dict, optional
        In-memory dictionary from `estimate_bwas(save_mode='none')`.
    common_snps_list : list or ndarray, optional
        List of SNPs to retain.
    m_annot : int, default 1173569
        Number of SNPs.
    chunk_size : int, default 500
        Number of voxels per chunk.
    n_blocks : int, default 200
        Number of jackknife blocks.
    twostep : int, default 30
        Two-step cutoff for LDSC.
    device : str, default 'cuda:0'
        Device to use.
    target_snp_col : str, default 'SNP'
        Column name for SNP IDs in the target GWAS file.
    target_z_col : str, default 'Z'
        Column name for Z-scores in the target GWAS file.
    target_n_col : str, default 'N'
        Column name for sample sizes in the target GWAS file.

    Returns
    -------
    rg : ndarray
        Voxel-wise genetic correlation.
    rg_se : ndarray
        Standard error of voxel-wise genetic correlation.
    gencov : ndarray
        Voxel-wise genetic covariance.
    gencov_se : ndarray
        Standard error of voxel-wise genetic covariance.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parameter validation
    if bwas_dict is None:
        if gwas_h5_template is None:
            raise ValueError("Must provide either `gwas_h5_template` or `bwas_dict`.")
        if '{chr}' not in gwas_h5_template:
            raise ValueError("`gwas_h5_template` must contain a '{chr}' placeholder.")
        print("Data source: HDF5 files.")
    else:
        print("Data source: In-memory BWAS dictionary.")

    if not os.path.exists(target_gwas_path):
        raise FileNotFoundError(f"Target GWAS file not found: {target_gwas_path}")
    else:
        print(f"Loading target GWAS summary statistics from: {target_gwas_path}")
        target_df = pd.read_csv(target_gwas_path, sep='\t')
    
    # Build a dictionary for faster matching
    target_snp_to_idx = {snp: idx for idx, snp in enumerate(target_df[target_snp_col].values)}

    # 1. Collect LD Scores, match SNPs, and compute target scalar h2
    print(f"Step 1: Collecting LD scores and computing target h2...")
    
    chr_matched_info = {}
    chr_h5_files = {}
    all_ld_scores = []
    all_n_voxel = []
    all_target_z = []
    all_n_target = []

    for chr_num in chromosome_range:
        ld_file = os.path.join(ld_score_folder, f'{chr_num}.l2.ldscore.gz')
        if not os.path.exists(ld_file):
            raise FileNotFoundError(f"LD score file not found: {ld_file}")
            
        ld_df = pd.read_csv(ld_file, sep='\t', compression='gzip')
        
        if bwas_dict is None:
            h5_file = gwas_h5_template.format(chr=chr_num)
            if not os.path.exists(h5_file):
                raise ValueError(f"HDF5 file not found: {h5_file}")
            gwas_snplist = load_h5_snplist(h5_file, chr_num)
            # Get sample size
            with h5py.File(h5_file, 'r') as h5f:
                n_samples = len(h5f['sampleid'])
            chr_h5_files[chr_num] = h5_file
        else:
            if chr_num not in bwas_dict:
                raise KeyError(f"Chromosome {chr_num} not found in `bwas_dict`.")
            gwas_snplist = bwas_dict[chr_num]['snplist']
            n_samples_voxel = len(bwas_dict[chr_num]['sampleid'])

        # Match LD score, HDF5, and Target GWAS SNPs
        snp_to_idx_h5 = {snp: idx for idx, snp in enumerate(gwas_snplist)}
        matched_indices = []
        matched_ld = []
        matched_target_z = []
        matched_target_n = []
        
        for i, snp in enumerate(ld_df['SNP'].values):
            if snp in snp_to_idx_h5 and snp in target_snp_to_idx:
                # Match HDF5 index
                h5_idx = snp_to_idx_h5[snp]
                matched_indices.append(h5_idx)
                # Match LD score
                matched_ld.append(ld_df['L2'].values[i])
                # Match Target GWAS data
                t_idx = target_snp_to_idx[snp]
                matched_target_z.append(target_df[target_z_col].values[t_idx])
                matched_target_n.append(target_df[target_n_col].values[t_idx])

        if len(matched_indices) == 0:
            raise ValueError(f"No valid SNPs matched across LD, HDF5, and Target GWAS for Chromosome {chr_num}.")

        chr_matched_info[chr_num] = np.array(matched_indices)
        all_ld_scores.extend(matched_ld)
        all_n_voxel.extend(n_samples_voxel * np.ones(len(matched_indices)))
        all_target_z.extend(matched_target_z)
        all_n_target.extend(matched_target_n)

    ld_all = np.array(all_ld_scores)
    n_voxel_all = np.array(all_n_voxel)
    target_z_all = np.array(all_target_z)
    target_n_all = np.array(all_n_target)

    print(f"Total matched SNPs: {len(ld_all)}")

    # Compute target global h2
    target_z_tensor = torch.from_numpy(target_z_all).to(device)
    target_chi2 = target_z_tensor ** 2
    target_chi2[torch.isnan(target_chi2)] = 1e-8
    n_target_tensor = torch.from_numpy(target_n_all).to(device=device, dtype=torch.float32)
    ld_tensor = torch.from_numpy(ld_all).to(device)
    n_voxel_tensor = torch.from_numpy(n_voxel_all).to(device)

    ldsc_target = Hsq(
        y=target_chi2.unsqueeze(0), w=ld_tensor, N=n_target_tensor, M=m_annot,
        n_blocks=n_blocks, intercept=1, twostep=twostep
    )
    h2_target = torch.clamp(ldsc_target.cat, 0, 1)
    print(f"Target h2: {h2_target.item():.4f}")

    # 2. Block-wise computation of Gencov and RG
    first_chr = list(chr_matched_info.keys())[0]
    if bwas_dict is None:
        with h5py.File(chr_h5_files[first_chr], 'r') as h5f:
            V = h5f.attrs['n_voxels']
    else:
        blocks = bwas_dict[first_chr]['blocks']
        V = max([int(k.split('_')[-1]) for k in blocks.keys()])

    print(f"Total voxels to process: {V}")

    gencov_voxel = np.zeros(V)
    gencov_se_voxel = np.zeros(V)
    rg_voxel = np.zeros(V)
    rg_se_voxel = np.zeros(V)

    print(f"Step 2: Block-wise processing (chunk_size={chunk_size})...")
    for start_idx in range(0, V, chunk_size):
        end_idx = min(start_idx + chunk_size, V)
        print(f"  -> Processing Voxels {start_idx} to {end_idx}...")

        z_voxel_list = []
        block_key = f'block_{start_idx}_{end_idx}'
        
        for chr_num, matched_indices in chr_matched_info.items():
            if bwas_dict is None:
                h5_file = chr_h5_files[chr_num]
                with h5py.File(h5_file, 'r') as h5f:
                    if block_key in h5f:
                        z_chunk = h5f[block_key][matched_indices].astype(np.float32)
                        z_voxel_list.append(torch.from_numpy(z_chunk).to(device))
                    else:
                        raise KeyError(f"Missing data block '{block_key}' in HDF5 file: {h5_file}.")
            else:
                if block_key in bwas_dict[chr_num]['blocks']:
                    z_matrix = bwas_dict[chr_num]['blocks'][block_key]
                    z_chunk = z_matrix[matched_indices, :].astype(np.float32)
                    z_voxel_list.append(torch.from_numpy(z_chunk).to(device))
                else:
                    raise KeyError(f"Missing data block '{block_key}' in bwas_dict for Chr {chr_num}.")

        z_voxel_chunk = torch.cat(z_voxel_list, dim=0)
        z_voxel_chunk[torch.isnan(z_voxel_chunk)] = 1e-8
        
        # Calculate voxel h2 (for gencov denominator)
        chi2_chunk = (z_voxel_chunk ** 2).T
        ldsc_voxel = Hsq(
            y=chi2_chunk, w=ld_tensor, N=n_voxel_tensor, M=m_annot,
            n_blocks=n_blocks, intercept=1, twostep=twostep
        )
        h2_chunk = torch.clamp(ldsc_voxel.cat, 0, 1)

        # Calculate gencov
        gencov_obj = Gencov(
            z1=z_voxel_chunk.T, z2=target_z_tensor,
            w=ld_tensor, N1=n_voxel_tensor, N2=n_target_tensor,
            M=m_annot, hsq1=h2_chunk.reshape(-1, 1),
            hsq2=h2_target, intercept_hsq1=1, intercept_hsq2=1,
            n_blocks=n_blocks, intercept_gencov=0, twostep=twostep
        )
        gencov_chunk = gencov_obj.cat
        gencov_se_chunk = gencov_obj.cat_se

        gencov_voxel[start_idx:end_idx] = gencov_chunk.cpu().numpy()
        gencov_se_voxel[start_idx:end_idx] = gencov_se_chunk.cpu().numpy()

        # Calculate rg and its standard error
        rg_chunk = gencov_chunk / torch.sqrt(torch.clamp(h2_chunk, 1e-8, 1) * h2_target)
        rg_chunk = torch.clamp(rg_chunk, -1, 1)
        
        denom_del = torch.sqrt(
            ldsc_target.tot_delete_values.reshape(1, -1).to(device) * 
            ldsc_voxel.tot_delete_values
        )
        rg_obj = RatioJackknife(rg_chunk, gencov_obj.tot_delete_values, denom_del)
        
        rg_voxel[start_idx:end_idx] = rg_chunk.cpu().numpy()
        rg_se_voxel[start_idx:end_idx] = rg_obj.jknife_se.cpu().numpy()

        print(f"     Block done. Mean rg: {np.nanmean(rg_chunk.cpu().numpy()):.4f} | Max rg: {np.nanmax(rg_chunk.cpu().numpy()):.4f} | Min rg: {np.nanmin(rg_chunk.cpu().numpy()):.4f}")

        del z_voxel_list, z_voxel_chunk, chi2_chunk, ldsc_voxel, gencov_obj, rg_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("All voxels processed successfully.")

    return rg_voxel, rg_se_voxel, gencov_voxel, gencov_se_voxel