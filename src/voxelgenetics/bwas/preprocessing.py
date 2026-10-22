import numpy as np
import torch
import nibabel as nib


def nets_zscore(x):
    """
    Z-score normalization along columns (features).
    
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_features)
        Input feature matrix.
        
    Returns
    -------
    x_zscore : ndarray, shape (n_samples, n_features)
        Z-scored feature matrix.
    """
    x_zscore = (x - x.mean(axis=0))
    stds = x.std(axis=0)
    
    # Handle zero-variance columns by setting std to 0.1
    index = stds == 0
    if np.sum(index) > 0:
        stds[index] = 0.1
        # In a real library, use logging.warning instead of print
        print(f'Warning: {np.sum(index)} of the features are all zero or constants. Normalizing them to all zeros...')
    
    x_zscore = x_zscore / stds
    return x_zscore


def g_impute_nan_as_mean(x):
    """
    Impute NaNs with column means. Supports both NumPy arrays and PyTorch tensors.
    
    Parameters
    ----------
    x : ndarray or torch.Tensor, shape (n_samples, n_features)
        Input matrix with potential NaNs.
        
    Returns
    -------
    x : ndarray or torch.Tensor
        Imputed matrix.
    """
    if torch.is_tensor(x):
        col_means = torch.nanmean(x, dim=0)
        for i in range(x.shape[1]):
            mask = torch.isnan(x[:, i])
            if mask.any():
                x[mask, i] = col_means[i]
    else:
        x_mean = np.nanmean(x, axis=0)
        for i in range(x.shape[1]):
            x[np.isnan(x[:, i]), i] = x_mean[i]
    return x


def get_res(X, Y):
    """
    Regress out covariates X from Y.
    
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_covariates)
        Covariate matrix.
    Y : ndarray, shape (n_samples, n_features)
        Target matrix.
        
    Returns
    -------
    Res : ndarray, shape (n_samples, n_features)
        Residualized Y.
    """
    beta = np.dot(np.linalg.pinv(X), Y)
    Res = Y - np.dot(X, beta)
    return Res


def get_data_parallel_new(
    i, 
    image_file_list, 
    image_mask, 
    modality, 
    network_index=0, 
    rh_file_list=None
):
    """
    Read image data for a single subject (supports NIfTI, MGH, and DU NPY formats).
    
    Parameters
    ----------
    i : int
        Index of the subject in `image_file_list`.
    image_file_list : list or ndarray
        List of image file paths for the primary hemisphere (or whole brain).
    image_mask : ndarray
        Brain mask for NIfTI data.
    modality : str
        Modality type: 'nifti', 'du', or 'freesurfer'.
    network_index : int, default 0
        Index for DU network data.
    rh_file_list : list or ndarray, optional
        List of image file paths for the right hemisphere (required if modality='freesurfer').
        
    Returns
    -------
    dat : ndarray
        Flattened 1D image data.
    """
    modality = modality.lower()
    
    if modality == 'du':
        dat = np.load(image_file_list[i])[:, network_index].flatten()
        
    elif modality == 'freesurfer':
        if rh_file_list is None:
            raise ValueError("`rh_file_list` must be provided when modality='freesurfer'.")
        # Load left hemisphere
        lh_data = image_file_list[i]
        lh_data1 = nib.load(lh_data).get_fdata().flatten()
        # Load right hemisphere
        rh_data = rh_file_list[i]
        rh_data1 = nib.load(rh_data).get_fdata().flatten()
        dat = np.concatenate((lh_data1, rh_data1))
        
    elif modality == 'nifti':
        info = nib.load(image_file_list[i])
        dat = info.get_fdata()[image_mask > 0]
        
    else:
        raise ValueError(f"Unknown modality: {modality}. Expected 'nifti', 'du', or 'freesurfer'.")
        
    return dat

