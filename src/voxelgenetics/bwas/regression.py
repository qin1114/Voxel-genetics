import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
from .preprocessing import g_impute_nan_as_mean, get_res, nets_zscore


def _perform_batch_bwas(burden_batch, Y_chunk, cov_expanded, n_cov, device):
    """Internal function to perform BWAS for a batch of SNPs."""
    n_samples = burden_batch.shape[0]
    actual_n_cov = cov_expanded.shape[-1]
    
    X_batch = burden_batch.unsqueeze(2)
    cov_batch = cov_expanded.expand(-1, X_batch.shape[1], -1)
    design_batch = torch.cat([X_batch, cov_batch], dim=2)
    
    design_batch_t = design_batch.permute(1, 2, 0)
    XtX_batch = torch.bmm(design_batch_t, design_batch_t.transpose(1, 2))
    
    ss_batch = torch.linalg.inv(XtX_batch)
    ss_Xt_batch = torch.bmm(ss_batch, design_batch_t)
    
    contrast_batch = torch.zeros((X_batch.shape[1], 1 + actual_n_cov, 1), device=device)
    contrast_batch[:, 0, 0] = 1.0
    
    tmp2_batch = torch.bmm(contrast_batch.transpose(1, 2), torch.bmm(ss_batch, contrast_batch))
    tmp2_sqrt_batch = torch.sqrt(tmp2_batch.squeeze(-1))
    
    Y_chunk_batch = Y_chunk.T.unsqueeze(0).expand(X_batch.shape[1], -1, -1).permute(0, 2, 1)
    beta_batch = torch.bmm(ss_Xt_batch, Y_chunk_batch)
    beta_X_batch = beta_batch[:, 0, :]
    
    predictions_batch = torch.bmm(design_batch_t.permute(0, 2, 1), beta_batch)
    residuals_batch = Y_chunk_batch - predictions_batch
    
    df = n_samples - (1 + n_cov)
    sigma_sq_batch = torch.sum(residuals_batch ** 2, dim=1) / df
    sigma_batch = torch.sqrt(sigma_sq_batch)
    
    t_stats_batch = beta_X_batch / (sigma_batch * tmp2_sqrt_batch)
    
    return t_stats_batch, beta_X_batch


def estimate_bwas(
    genotype_bed,
    imaging_matrix,
    sampleid,
    snplist,
    output_prefix=None,

    covariates=None,
    residualize_y=False,
    include_covariates_in_model=False,
    n_cov=1,

    burden_threshold_ratio=0.01,
    chunk_size=500,
    burden_batch_size=100,
    device='cuda:0',
    
    save_mode='h5',
    npy_t_threshold=3.29,
    npy_min_sig_voxels=500,
):
    """
    Run BWAS.
    
    Returns
    -------
    h5_file (str) if save_mode == 'h5'
    npy_dir (str) if save_mode == 'npy'
    dict if save_mode == 'none' (Keys are 'block_start_end', values are numpy arrays of shape (n_snps, n_voxels_in_chunk))
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    n_samples, n_snps = genotype_bed.shape
    _, n_voxels = imaging_matrix.shape
    
    if save_mode not in ['none', 'h5', 'npy']:
        raise ValueError("`save_mode` must be 'none', 'h5', or 'npy'.")

    # Check if covariates or n_cov are provided when not regressing internally
    if (not residualize_y) and (not include_covariates_in_model):
        if covariates is not None:
            n_cov = covariates.shape[1]
        else:
            if n_cov is None:
                raise ValueError(
                    "Covariates must be considered when performing BWAS! "
                    "If you have pre-regressed covariates, please provide `n_cov`. "
                    "Otherwise, provide `covariates` or set `residualize_y=True` `include_covariates_in_model=True`."
                )

    # 1. Preprocessing (regress out covariates if required)
    if residualize_y:
        if covariates is not None:
            print("Residualizing imaging matrix with covariates on CPU...")
            n_cov = covariates.shape[1] + 1
            covariates_z = nets_zscore(g_impute_nan_as_mean(covariates))
            imaging_matrix = get_res(covariates_z, imaging_matrix)
        else:
            raise ValueError("`covariates` must be provided when `residualize_y=True`.")
    
    Y_cpu = torch.FloatTensor(imaging_matrix)
    
    # 2. Determine Covariate Matrix for the Model
    if include_covariates_in_model:
        if covariates is not None:
            print("Including covariates while performing BWAS...")
            n_cov = covariates.shape[1] + 1
            cov_gpu = torch.FloatTensor(nets_zscore(g_impute_nan_as_mean(covariates))).to(device)
        else:
            raise ValueError("`covariates` must be provided when `include_covariates_in_model=True`.")
    else:
        cov_gpu = torch.ones(n_samples, 1).to(device)
        
    cov_expanded = cov_gpu.unsqueeze(1)

    # 3. Filter Burdens
    threshold = n_samples * burden_threshold_ratio
    burden_train_scores = torch.FloatTensor(genotype_bed).to(device)
    n_burden_counts = torch.sum(burden_train_scores > 0, dim=0)
    valid_burdens_mask = n_burden_counts >= threshold
    
    valid_snps = valid_burdens_mask.sum().item()
    burden_train_scores = burden_train_scores[:, valid_burdens_mask]
    burden_train_scores = g_impute_nan_as_mean(burden_train_scores)
    
    valid_snp_indices = torch.where(valid_burdens_mask)[0].cpu().numpy()
    snplist_valid = snplist[valid_burdens_mask]

    # =========================================================================
    # Branch A：save_mode == 'h5'
    # =========================================================================
    if save_mode == 'h5':
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        h5_file = f"{output_prefix}.h5"
        
        block_size = chunk_size
        n_blocks = (n_voxels + block_size - 1) // block_size
        
        with h5py.File(h5_file, 'w') as h5f:
            h5f.attrs['n_snps'] = valid_snps
            h5f.attrs['n_voxels'] = n_voxels
            h5f.attrs['chunk_size'] = chunk_size
            h5f.attrs['n_blocks'] = n_blocks

            h5f.create_dataset('snplist', data=snplist_valid, dtype=h5py.special_dtype(vlen=str))
            h5f.create_dataset('sampleid', data=sampleid, compression='gzip')
            
            for y_start in tqdm(range(0, n_voxels, chunk_size), desc="Processing Voxel Chunks"):
                y_end = min(y_start + chunk_size, n_voxels)
                Y_chunk = Y_cpu[:, y_start:y_end].to(device)
                
                t_chunk_cpu = torch.zeros((valid_snps, y_end - y_start))
                
                for b_start in range(0, valid_snps, burden_batch_size):
                    b_end = min(b_start + burden_batch_size, valid_snps)
                    burden_batch = burden_train_scores[:, b_start:b_end]
                    
                    t_batch, _ = _perform_batch_bwas(burden_batch, Y_chunk, cov_expanded, n_cov, device)
                    t_chunk_cpu[b_start:b_end, :] = t_batch.cpu()
                    
                    del t_batch, burden_batch
                
                block_key = f'block_{y_start}_{y_end}'
                h5f.create_dataset(block_key, data=t_chunk_cpu.numpy(), dtype=np.float16)
                
                del Y_chunk, t_chunk_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return h5_file

    # =========================================================================
    # Branch B：save_mode == 'npy'
    # =========================================================================
    elif save_mode == 'npy':
        npy_dir = f"{output_prefix}"
        os.makedirs(npy_dir, exist_ok=True)
        
        for b_start in tqdm(range(0, valid_snps, burden_batch_size), desc="Processing SNPs"):
            b_end = min(b_start + burden_batch_size, valid_snps)
            burden_batch = burden_train_scores[:, b_start:b_end]
            current_batch_size = b_end - b_start
            
            t_batch_gpu = torch.zeros((current_batch_size, n_voxels), device=device)

            for y_start in range(0, n_voxels, chunk_size):
                y_end = min(y_start + chunk_size, n_voxels)
                Y_chunk = Y_cpu[:, y_start:y_end].to(device)
                
                t_batch, _ = _perform_batch_bwas(burden_batch, Y_chunk, cov_expanded, n_cov, device)
                t_batch_gpu[:, y_start:y_end] = t_batch
                
                del t_batch, Y_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            sig_counts = torch.sum(torch.abs(t_batch_gpu) > npy_t_threshold, dim=1)
            for i in range(current_batch_size):
                if sig_counts[i] >= npy_min_sig_voxels:
                    snp_original_idx = valid_snp_indices[b_start + i]
                    np.save(f"{npy_dir}/snp_{snp_original_idx}_tstat.npy", t_batch_gpu[i].cpu().numpy())

            del t_batch_gpu, burden_batch

        return npy_dir

    # =========================================================================
    # Branch C：save_mode == 'none'
    # =========================================================================
    else:
        results_dict = {}
        results_dict['blocks'] = {}
        
        for y_start in tqdm(range(0, n_voxels, chunk_size), desc="Processing Voxel Chunks"):
            y_end = min(y_start + chunk_size, n_voxels)
            Y_chunk = Y_cpu[:, y_start:y_end].to(device)
            
            t_chunk_cpu = torch.zeros((valid_snps, y_end - y_start))
            
            for b_start in range(0, valid_snps, burden_batch_size):
                b_end = min(b_start + burden_batch_size, valid_snps)
                burden_batch = burden_train_scores[:, b_start:b_end]
                
                t_batch, _ = _perform_batch_bwas(burden_batch, Y_chunk, cov_expanded, n_cov, device)
                t_chunk_cpu[b_start:b_end, :] = t_batch.cpu()
                
                del t_batch, burden_batch
            
            block_key = f'block_{y_start}_{y_end}'
            results_dict['blocks'][block_key] = t_chunk_cpu.numpy()
            results_dict['snplist'] = snplist_valid
            results_dict['sampleid'] = sampleid
            
            del Y_chunk, t_chunk_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results_dict
