from scipy.stats import norm
import numpy as np
from scipy import special, ndimage, stats
import pandas as pd
import nibabel as nib


def estimate_fwhm_3d(fMRI):
    """
    Estimate FWHM per time point from a 4D image.

    Parameters
    ----------
    fMRI : ndarray, shape (n1, n2, n3, n4)

    Returns
    -------
    fwhm : ndarray, shape (n4,)
    """
    n1, n2, n3, n4 = fMRI.shape

    # Propagate NaNs (equivalent to the original expression)
    fMRI = fMRI * fMRI / fMRI

    dx = np.diff(fMRI, 1, 0)
    varX = np.nanvar(np.reshape(dx, [(n1 - 1) * n2 * n3, n4]), axis=0)

    dy = np.diff(fMRI, 1, 1)
    varY = np.nanvar(np.reshape(dy, [n1 * (n2 - 1) * n3, n4]), axis=0)

    dz = np.diff(fMRI, 1, 2)
    varZ = np.nanvar(np.reshape(dz, [n1 * n2 * (n3 - 1), n4]), axis=0)

    varXYZ = (varX * varY * varZ) ** (1 / 3.0)

    varImg = np.zeros((n4,))
    for i in range(n4):
        tmp = fMRI[:, :, :, i]
        varImg[i] = np.nanvar(tmp)

    fwhm = np.real(np.sqrt(-2 * np.log(2) / np.log(1 - varXYZ / 2 / varImg)))
    return fwhm


def _grf_3d_ec_density(t):
    """
    Euler characteristic (EC) density of a 3D Gaussian random field.

    Parameters
    ----------
    t : float
        Threshold.

    Returns
    -------
    EC : ndarray, shape (4,)
        EC densities for dimensions 0, 1, 2, 3.
    """
    EC = np.zeros((4,))

    a = 4.0 * np.log(2.0)
    a1 = 2.0 * np.pi
    b = np.exp(-np.square(t) / 2.0)

    EC[0] = 1.0 - norm.cdf(t)
    EC[1] = a ** (1.0 / 2.0) / a1 * b
    EC[2] = a / (a1 ** (3.0 / 2.0)) * b * t
    EC[3] = a ** (3.0 / 2.0) / (a1 ** 2.0) * b * (np.power(t, 2) - 1)

    return EC


def cluster_p_3d(nvoxel, cdt_z, cluster_size, fwhm):
    """
    Cluster-extent p-values for a single cluster using the 3D GRF approximation.

    Parameters
    ----------
    nvoxel : int
        Number of voxels in the search space.
    cdt_z : float
        Cluster-defining threshold.
    cluster_size : int
        Cluster extent (number of voxels).
    fwhm : float or sequence of 3 floats
        Smoothness in mm.

    Returns
    -------
    p_corrected : float
        FWE-corrected cluster p-value.
    p_uncorrected : float
        Uncorrected cluster-extent p-value.
    """
    fwhm = np.array(fwhm, dtype="float64")
    cdt_z = np.array(cdt_z, dtype="float64")
    cluster_size = np.array(cluster_size, dtype="float64")

    r1 = (nvoxel / (4.0 / 3.0 * np.pi)) ** (1 / 3.0)
    R1 = [
        1,
        4 * (r1 ** 3 / np.prod(fwhm)) ** (1 / 3.0),
        2 * np.pi * (r1 ** 3 / np.prod(fwhm)) ** (2 / 3.0),
        nvoxel / np.prod(fwhm),
    ]

    V = nvoxel

    EC = _grf_3d_ec_density(cdt_z)

    EN = V * (norm.cdf(-np.abs(cdt_z)))

    EL1 = 0.0
    for i in range(4):
        EL1 = EL1 + R1[i] * EC[i]

    if EL1 < 0:
        p_uncorrected = 1.0
        p_corrected = 1.0
    else:
        phi = (special.gamma(3.0 / 2.0 + 1) * EL1 / EN) ** (2.0 / 3.0)
        p_uncorrected = np.exp(-phi * cluster_size ** (2.0 / 3.0))
        p_corrected = 1 - np.exp(-EL1 * p_uncorrected)

    return p_corrected, p_uncorrected


def cluster_correct_volume(
    stat_img,
    mask,
    cdt_z,
    fwhm=None,
    nvoxel=None,
    alpha=5e-8,
    atlas=None,
    affine=None,
    min_voxel_count=100,
    output_file=None,
    target_img=None,
    target_p_threshold=0.05,
):
    """
    One-stop 3D cluster-extent RFT correction.

    Parameters
    ----------
    stat_img : ndarray
        Statistical map with the same shape as `mask`.
    mask : ndarray of bool
        Search-space mask.
    cdt_z : float
        Cluster-defining threshold.
    fwhm : float or (fx, fy, fz), optional
        Smoothness in mm. If None, it is estimated from `stat_img`.
    nvoxel : int, optional
        Number of voxels in the search space. If None, uses `mask.sum()`.
    alpha : float, default 5e-8
        FWE significance threshold.
    atlas : tuple (atlas_data, region_dict), optional
        Used to annotate brain regions.
    min_voxel_count : int, default 100
        Minimum number of voxels a region must have within the cluster to be reported.
    affine : ndarray, shape (4, 4), optional
        If provided, returns MNI coordinates of each cluster peak.
    output_file : str, optional
        If provided, saves a NIfTI file containing only the significant clusters.
    target_img : ndarray, optional
        Target (replication) set statistical map. If provided, computes replication rates.
    target_p_threshold : float, default 0.05
        P-value threshold for defining a voxel as "replicated" in the target/abcd set (used for replication rate calculation).

    Returns
    -------
    pandas.DataFrame
        Columns: cluster_index, cluster_size, cluster_p,
        cluster_p_uncorrected, peak_stat, peak_voxel,
        peak_mni_x/y/z (when `affine` is given),
        brain_regions (when `atlas` is given).
    """
    if nvoxel is None:
        nvoxel = int(mask.sum())

    if fwhm is None:
        fwhm_arr = estimate_fwhm_3d(stat_img[..., None])
        fwhm3 = np.repeat(fwhm_arr, 3)
    else:
        fwhm3 = np.broadcast_to(np.asarray(fwhm, float), (3,))

    # Threshold and find connected components
    thresh = (np.abs(stat_img) > cdt_z) * mask
    labeled, n_features = ndimage.label(thresh)

    rows = []
    for i in range(1, n_features + 1):
        cluster_mask = labeled == i
        size = int(cluster_mask.sum())
        p_corr, p_unc = cluster_p_3d(nvoxel, cdt_z, size, fwhm3)
        if p_corr >= alpha:
            # Keep only significant clusters
            continue

        vals = stat_img[cluster_mask]
        peak_local = int(np.argmax(np.abs(vals)))
        idx = np.where(cluster_mask)
        peak_vox = np.array(
            [idx[0][peak_local], idx[1][peak_local], idx[2][peak_local]]
        )

        row = dict(
            cluster_index=i,
            cluster_size=size,
            cluster_p=p_corr,
            cluster_p_uncorrected=p_unc,
            peak_stat=float(vals[peak_local]),
            peak_voxel=tuple(peak_vox),
        )

        if affine is not None:
            from nibabel.affines import apply_affine

            mni = apply_affine(affine, peak_vox.reshape(1, 3))[0]
            row.update(
                peak_mni_x=float(mni[0]),
                peak_mni_y=float(mni[1]),
                peak_mni_z=float(mni[2]),
            )

        if target_img is not None:
            target_cluster_values = target_img[cluster_mask]
            discovery_cluster_values = stat_img[cluster_mask]
            p_less_than_threshold = np.sum((np.abs(target_cluster_values) > stats.norm.ppf(1 - target_p_threshold/2)) & (np.sign(target_cluster_values) == np.sign(discovery_cluster_values)))
            total_voxels = len(target_cluster_values)
            if total_voxels > 0:
                proportion = p_less_than_threshold / total_voxels
            else:
                proportion = 0
            row.update(
                target_proportion=proportion,
            )

        if atlas is not None:
            from ..atlas.regions import region_summary

            row["brain_regions"] = region_summary(
                cluster_mask=cluster_mask,
                atlas_data=atlas[0],
                region_dict=atlas[1],
                stat_img=stat_img,
                target_img=target_img,
                p_threshold=target_p_threshold,
                min_voxel_count=min_voxel_count,
            )
        rows.append(row)

    
    df = pd.DataFrame(rows)
    # Save NIfTI if requested
    if output_file is not None:
        if len(df) > 0:
            sig_labels = df["cluster_index"].values
            sig_mask = np.isin(labeled, sig_labels)
            out_data = np.where(sig_mask, stat_img, 0).astype(np.float32)
            
            # Use provided affine, otherwise fallback to identity
            if affine is None:
                raise ValueError("`affine` is required to ensure correct spatial orientation for MNI coordinates and output files.")
            else:
                save_affine = affine
            out_img = nib.Nifti1Image(out_data, save_affine)
            nib.save(out_img, output_file)
            print(f"Saved {len(sig_labels)} significant clusters to {output_file}")

        else:
            print(f'Find no significant cluster!')

    return df