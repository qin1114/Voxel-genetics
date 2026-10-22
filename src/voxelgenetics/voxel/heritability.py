"""
voxelgenetics.voxel.heritability
--------------------------------
Voxel-wise heritability estimation using LDSC and PyTorch (GPU-accelerated).
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch

from ..ldsc.regression import Hsq
from .h5_io import load_h5_snplist


def estimate_h2_voxel(
    chromosome_range,
    ld_score_folder,
    gwas_h5_template,
    bwas_dict=None,
    common_snps_list=None,
    m_annot=1173569,
    chunk_size=500,
    n_blocks=200,
    twostep=30,
    device='cuda:0',
):
    """
    Estimate voxel-wise heritability (h2) directly from chunked HDF5 files.

    Parameters
    ----------
    chromosome_range : iterable
        List of chromosome numbers (e.g., range(1, 23)).
    ld_score_folder : str
        Directory containing LD score files (e.g., '{chr}.l2.ldscore.gz').
    gwas_h5_template : str
        Full path template of the voxel GWAS HDF5 files, with a ``{chr}``
        placeholder that will be filled with each chromosome number, e.g.
        ``'/path/to/VBM_2mm_s_chr{chr}_test.h5'``.
    bwas_dict : dict, optional
        Output dictionary from `estimate_bwas(save_mode='none')`. 
        If provided, `gwas_h5_template` will be ignored, and data will be read from this dictionary instead.
    common_snps_list : list or ndarray, optional
        List of SNPs to retain. If None, all SNPs in the LD score files are used.
    m_annot : int, default 1173569
        Number of SNPs used for LDSC intercept.
    chunk_size : int, default 500
        Number of voxels to process at once.
    n_blocks : int, default 200
        Number of jackknife blocks.
    twostep : int, default 30
        Two-step cutoff for LDSC.
    device : str, default 'cuda:0'
        Device to use for computation ('cuda:0' or 'cpu').

    Returns
    -------
    h2 : ndarray
        Voxel-wise heritability.
    h2_se : ndarray
        Standard error of voxel-wise heritability.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if bwas_dict is None:
        if gwas_h5_template is None:
            raise ValueError("Must provide either `gwas_h5_template` or `bwas_dict`.")
        if '{chr}' not in gwas_h5_template:
            raise ValueError(
                "`gwas_h5_template` must contain a '{chr}' placeholder, "
                f"got: {gwas_h5_template}"
            )
        print("Data source: HDF5 files.")
    else:
        print("Data source: In-memory BWAS dictionary.")

    # 1. Collect LD scores and match SNPs
    all_ld_scores = []
    all_n = []
    chr_matched_info = {}
    chr_h5_files = {}  # chr_num -> resolved h5 path

    print(f"Step 1: Collecting LD scores and matching SNPs for {len(list(chromosome_range))} chromosomes...")
    
    for chr_num in chromosome_range:
        ld_file = os.path.join(ld_score_folder, f'{chr_num}.l2.ldscore.gz')
        if not os.path.exists(ld_file):
            raise FileNotFoundError(
                f"LD score file not found for Chromosome {chr_num}. "
                f"Expected path: {ld_file}"
            )
        else:
            ld_df = pd.read_csv(ld_file, sep='\t', compression='gzip')
            ld_score = ld_df['L2'].values
            ld_snp_list = ld_df['SNP'].values

            if common_snps_list is not None:
                common_set = set(common_snps_list)
                ld_df = ld_df[ld_df['SNP'].isin(common_set)]
                if len(ld_df) == 0:
                    raise ValueError(
                        f"No common SNPs found between LD score and HDF5 for Chromosome {chr_num}. "
                        f"Please check if the SNP naming conventions or reference panels are consistent."
                    )

        if bwas_dict is None:
            # Resolve HDF5 path for this chromosome
            h5_file = gwas_h5_template.format(chr=chr_num)
            if not os.path.exists(h5_file):
                raise ValueError(f"No HDF5 files found matching pattern: {gwas_h5_template.format(chr='*')}")
            gwas_snplist = load_h5_snplist(h5_file, chr_num)
            # Get sample size
            with h5py.File(h5_file, 'r') as h5f:
                n_samples = len(h5f['sampleid'])

        else:
            gwas_snplist = bwas_dict[chr_num]['snplist']
            n_samples = len(bwas_dict[chr_num]['sampleid'])

        snp_to_idx = {snp: idx for idx, snp in enumerate(gwas_snplist)}

        ## Match between LD scores file and snp file
        matched_indices = []
        matched_ld = []

        for i, snp in enumerate(ld_snp_list):
            if snp in snp_to_idx:
                matched_indices.append(snp_to_idx[snp])
                matched_ld.append(ld_score[i])

        if len(matched_indices) == 0:
            raise ValueError(f"No valid SNPs matched between LD scores and HDF5 files for Chromosome {chr_num}.")

        print(f"  - Chr {chr_num}: Matched {len(matched_indices)} SNPs.")

        chr_matched_info[chr_num] = np.array(matched_indices)
        all_ld_scores.extend(matched_ld)
        all_n.extend(n_samples * np.ones(len(matched_indices)))

    ld_all = np.array(all_ld_scores)
    n_all = np.array(all_n)
    
    print(f"Step 1 Finished. Total matched SNPs across all chromosomes: {len(ld_all)}")

    # Get total number of voxels
    first_chr = list(chr_matched_info.keys())[0]
    if bwas_dict is None:
        with h5py.File(chr_h5_files[first_chr], 'r') as h5f:
            V = h5f.attrs['n_voxels']
    else:
        first_block_key = list(bwas_dict[first_chr]['blocks'].keys())[0]
        V = sum([b.shape[1] for b in bwas_dict[first_chr]['blocks'].values()])
    
    print(f"Total voxels to process: {V}")

    h2_voxel = np.zeros(V)
    h2_se_voxel = np.zeros(V)

    ld_tensor = torch.from_numpy(ld_all).to(device)
    n_tensor = torch.from_numpy(n_all).to(device)

    # 2. Block-wise processing
    print(f"Step 2: Block-wise processing (chunk_size={chunk_size})...")
    for start_idx in range(0, V, chunk_size):
        end_idx = min(start_idx + chunk_size, V)
        
        print(f"  -> Processing Voxels {start_idx} to {end_idx}...")

        # Get voxel-level z scores from h5 file
        z_voxel_list = []
        for chr_num, matched_indices in chr_matched_info.items():
            if bwas_dict is None:
                h5_file = chr_h5_files[chr_num]
                with h5py.File(h5_file, 'r') as h5f:
                    block_key = f'block_{start_idx}_{end_idx}'
                    if block_key in h5f:
                        z_chunk = h5f[block_key][matched_indices].astype(np.float32)
                        z_voxel_list.append(torch.from_numpy(z_chunk).to(device))
                    else:
                        raise KeyError(
                            f"Missing data block '{block_key}' in HDF5 file: {h5_file}. "
                            f"Please check if the HDF5 file is corrupted or was generated with a different chunk size.")
            else:
                block_key = f'block_{start_idx}_{end_idx}'
                if block_key in bwas_dict[chr_num]['blocks'].keys():
                    z_matrix = bwas_dict[chr_num]['blocks'][block_key]
                    z_chunk = z_matrix[matched_indices, :].astype(np.float32)
                    z_voxel_list.append(torch.from_numpy(z_chunk).to(device))
                else:
                    raise KeyError(
                        f"Missing data block '{block_key}' in bwas_dict for Chromosome {chr_num}."
                    )

        z_voxel_chunk = torch.cat(z_voxel_list, dim=0)
        z_voxel_chunk[torch.isnan(z_voxel_chunk)] = 1e-8

        chi2_chunk = (z_voxel_chunk ** 2).T  # (chunk_size, n_snps)

        # Run LDSC
        ldsc_obj = Hsq(
            y=chi2_chunk,
            w=ld_tensor,
            N=n_tensor,
            M=m_annot,
            n_blocks=n_blocks,
            intercept=1,
            twostep=twostep,
        )

        h2_chunk = torch.clamp(ldsc_obj.cat, 0, 1)
        h2_se_chunk = ldsc_obj.cat_se

        h2_voxel[start_idx:end_idx] = h2_chunk.cpu().numpy()
        h2_se_voxel[start_idx:end_idx] = h2_se_chunk.cpu().numpy()
        
        print(f"     Block done. Mean h2: {np.mean(h2_chunk.cpu().numpy()):.4f} | Max h2: {np.max(h2_chunk.cpu().numpy()):.4f}")

        del z_voxel_list, z_voxel_chunk, chi2_chunk, ldsc_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("All voxels processed successfully.")

    return h2_voxel, h2_se_voxel