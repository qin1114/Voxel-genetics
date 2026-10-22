import numpy as np
from scipy import stats


def region_summary(
    cluster_mask,
    atlas_data,
    region_dict,
    stat_img,
    target_img=None,
    p_threshold=0.05,
    min_voxel_count=100,
):
    """
    Summarize the brain regions overlapped by a cluster and compute replication rates.

    Parameters
    ----------
    cluster_mask : ndarray of bool
        Boolean mask of the cluster (same shape as atlas_data).
    atlas_data : ndarray
        Atlas label map (e.g., JHU or UKB).
    region_dict : dict
        Mapping from integer label to region name.
    stat_img : ndarray
        Discovery set statistical map (used to determine sign of the effect).
    target_img : ndarray, optional
        Target (replication) set statistical map. If provided, computes target replication rate.
    p_threshold : float, default 0.05
        Two-sided p-value threshold for defining "replicated" voxels.
    min_voxel_count : int, default 100
        Minimum number of voxels a region must have within the cluster to be reported.

    Returns
    -------
    result_dict : dict
        Nested dictionary: {region_name: {region_name, voxel_rate, target_rate, ...}}
    """
    cluster_indices = np.where(cluster_mask)
    cluster_labels = atlas_data[cluster_indices]
    
    # Keep only valid atlas labels (>0)
    cluster_labels = cluster_labels[cluster_labels > 0]
    
    if len(cluster_labels) == 0:
        return {}
    
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    total_voxels_in_cluster = np.sum(counts)
    
    z_threshold = stats.norm.ppf(1 - p_threshold / 2)
    
    result_dict = {}
    for label, count in zip(unique_labels, counts):
        voxel_ratio = count / total_voxels_in_cluster
        
        if count >= min_voxel_count:
            region_name = region_dict.get(int(label), f"Region_{label}")
            
            # Create a mask for the intersection of this cluster and this specific region
            region_mask = (atlas_data == label) & cluster_mask
            
            # Extract values
            discovery_vals = stat_img[region_mask]
            
            target_rate = 0.0
            if target_img is not None:
                target_vals = target_img[region_mask]
                if len(target_vals) > 0:
                    significant = (np.abs(target_vals) > z_threshold) & (np.sign(target_vals) == np.sign(discovery_vals))
                    target_rate = float(np.sum(significant) / len(target_vals))

            result_dict[region_name] = {
                "region_name": region_name,
                "voxel_rate": float(voxel_ratio),
                "target_rate": target_rate,
            }
            
    return result_dict



def region_summary_freesurfer(
    cluster_vertices,
    annot_data,
    discovery_img,
    target_img=None,
    p_threshold=0.05,
    min_voxel_count=100,
):
    """
    Summarize the brain regions overlapped by a surface cluster and compute replication rates.

    Parameters
    ----------
    cluster_vertices : ndarray
        Indices of the vertices belonging to the cluster.
    annot_data : tuple (labels, region_dict)
        Output from `load_fs_annot`.
    discovery_img : ndarray
        Discovery set statistical map (1D array of vertices).
    target_img : ndarray, optional
        Target (replication) set statistical map.
    p_threshold : float, default 0.05
        Two-sided p-value threshold for defining "replicated" vertices.
    min_voxel_count : int, default 100
        Minimum number of voxels a region must have within the cluster to be reported.

    Returns
    -------
    dict
        Nested dictionary: {region_name: {region_name, voxel_rate, target_rate}}
    """
    if annot_data is None:
        return {}

    labels, region_dict = annot_data
    vertex_labels = labels[cluster_vertices]
    
    valid_mask = vertex_labels > 0
    valid_labels = vertex_labels[valid_mask]
    total = len(valid_labels)

    if total == 0:
        return {}

    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    result_dict = {}

    z_thresh = stats.norm.ppf(1 - p_threshold / 2)
    discovery_vals = discovery_img[cluster_vertices]

    for label, count in zip(unique_labels, counts):
        pct = count / total
        if (count >= min_voxel_count) and (label != 0):
            region_name = region_dict.get(label, f'Region_{label}')

            target_rate = None
            if target_img is not None:
                target_region_vals = target_img[cluster_vertices][vertex_labels == label]
                if len(target_region_vals) > 0:
                    sig_mask = (np.abs(target_region_vals) > z_thresh) & (np.sign(target_region_vals) == np.sign(discovery_vals[vertex_labels == label]))
                    target_rate = float(np.sum(sig_mask) / len(target_region_vals))

            result_dict[region_name] = {
                'region_name': region_name,
                'target_rate': target_rate,
                'voxel_rate': float(pct),
            }

    return result_dict