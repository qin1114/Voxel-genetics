"""
voxelgenetics.rft.surface
-------------------------

Random field theory correction for 2D cortical surfaces (FreeSurfer).
"""

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
from scipy.special import gamma
from nibabel.freesurfer.io import read_geometry, read_label, read_annot
import pandas as pd
import nibabel as nib


def load_fs_annot(annot_path):
    """
    Load a FreeSurfer .annot file and return labels and a region name dictionary.

    Parameters
    ----------
    annot_path : str
        Path to the .annot file (e.g., 'lh.aparc.annot').

    Returns
    -------
    labels : ndarray
        Array of integer labels for each vertex.
    region_dict : dict
        Mapping from integer label to region name.
    """
    labels, ctab, names = read_annot(annot_path)
    names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in names]

    region_dict = {i: name for i, name in enumerate(names)}
    return labels, region_dict


def build_surface_mesh(surface_file, cortex_label=None):
    """
    Build a FreeSurfer surface mesh for RFT correction.

    Parameters
    ----------
    surface_file : str
        Path to the surface geometry file (e.g., 'lh.white').
    cortex_label : str, optional
        Path to the cortex label file (e.g., 'lh.cortex.label').
        Used to define the search space (mask).

    Returns
    -------
    dict
        A dictionary containing mesh geometry (coords, faces, mask, edges,
        adjacency, vertex_area, area_mm2, boundary_length_mm, euler_char, n_vertices).
    """
    coords, faces = read_geometry(surface_file)
    coords = np.asarray(coords, float)
    faces = np.asarray(faces, int)
    n = len(coords)

    mask = np.ones(n, bool)
    if cortex_label is not None:
        mask[:] = False
        mask[read_label(cortex_label)] = True

    # Unique edges
    e = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    e.sort(axis=1)
    edges0 = np.unique(e, axis=0)
    edges = edges0[mask[edges0[:, 0]] & mask[edges0[:, 1]]]

    # Adjacency matrix
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    adj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    # Edge lengths
    edge_length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)

    # Vertex areas
    mf = faces[mask[faces].all(axis=1)]
    tri = coords[mf]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    vertex_area = np.zeros(n)
    np.add.at(vertex_area, mf.ravel(), np.repeat(tri_area / 3.0, 3))

    # Boundary and Euler characteristic
    te = np.vstack((mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]))
    te.sort(axis=1)
    ue, cnt = np.unique(te, axis=0, return_counts=True)
    be = ue[cnt == 1]
    boundary_length = np.linalg.norm(coords[be[:, 0]] - coords[be[:, 1]], axis=1).sum() if len(be) else 0.0
    chi = len(np.unique(mf.ravel())) - len(ue) + len(mf)

    return dict(
        coords=coords,
        faces=faces,
        mask=mask,
        edges=edges,
        edge_length=edge_length,
        adjacency=adj,
        vertex_area=vertex_area,
        area_mm2=tri_area.sum(),
        boundary_length_mm=boundary_length,
        euler_char=chi,
        n_vertices=n,
    )


def estimate_fwhm_surface(x, mesh):
    """
    Estimate smoothness (FWHM in mm) of a surface statistical map.
    """
    x = np.asarray(x, float).squeeze()
    mask = mesh["mask"] & np.isfinite(x)
    e = mesh["edges"]

    y = x.copy()
    y[mask] -= np.mean(y[mask])
    var_img = np.mean(y[mask] ** 2)

    ve = mask[e[:, 0]] & mask[e[:, 1]]
    ee = e[ve]
    diff = y[ee[:, 0]] - y[ee[:, 1]]

    if var_img <= 0 or not len(diff):
        return np.nan

    rho = np.clip(1.0 - np.mean(diff ** 2) / (2.0 * var_img), 1e-8, 1 - 1e-8)
    d_rms = np.sqrt(np.mean(mesh["edge_length"][ve] ** 2))
    return d_rms * np.sqrt(-2.0 * np.log(2.0) / np.log(rho))


def _grf_2d_ec_density(z):
    """2D Gaussian random field Euler characteristic density."""
    z = abs(float(z))
    a = 4.0 * np.log(2.0)
    b = np.exp(-z * z / 2.0)
    return np.array([
        norm.sf(z),
        np.sqrt(a) / (2.0 * np.pi) * b,
        a / (2.0 * np.pi) ** 1.5 * b * z,
    ])


def _surface_cluster_p(cluster_area, cdt_z, geometry, sidedness=2):
    """Compute cluster-extent FWE p-value for a surface cluster."""
    R, area = geometry["R"], geometry["area_mm2"]
    EC = _grf_2d_ec_density(cdt_z)
    EN = sidedness * area * norm.sf(abs(cdt_z))
    EL1 = sidedness * np.sum(R * EC)
    
    if EN <= 0 or EL1 <= 0:
        return 1.0, 1.0
        
    phi = gamma(2.0) * EL1 / EN
    p_unc = np.exp(-phi * cluster_area)
    p_fwe = 1.0 - np.exp(-EL1 * p_unc)
    return float(np.clip(p_fwe, 0, 1)), float(np.clip(p_unc, 0, 1))


def cluster_correct_surface(
    lh_stat,
    rh_stat,
    lh_mesh,
    rh_mesh,
    cdt_z=3.29,
    fwhm=None,
    alpha=5e-8,
    sign="abs",
    affine=None,
    lh_annot=None,
    rh_annot=None,
    min_voxel_count=100,
    output_prefix=None,
    lh_target=None,
    rh_target=None,
    target_p_threshold=0.05,
):
    """
    One-stop cluster-extent RFT correction for FreeSurfer surfaces.

    Parameters
    ----------
    lh_stat : ndarray
        Left hemisphere statistical map.
    rh_stat : ndarray
        Right hemisphere statistical map.
    lh_mesh : dict
        Left hemisphere mesh dictionary from `build_surface_mesh`.
    rh_mesh : dict
        Right hemisphere mesh dictionary from `build_surface_mesh`.
    cdt_z : float, default 3.29
        Cluster-defining threshold.
    fwhm : float or (fx, fy, fz), optional
        Smoothness in mm. If None, it is estimated from `stat_img`.
    alpha : float, default 5e-8
        FWE significance threshold.
    sign : str, default "abs"
        Which clusters to search for: "pos", "neg", or "abs" (both).
    affine : ndarray, shape (4, 4), optional
        Needed for mgh output.
    lh_annot : tuple (labels, region_dict), optional
        Left hemisphere annotation data from `load_fs_annot`.
    rh_annot : tuple (labels, region_dict), optional
        Right hemisphere annotation data from `load_fs_annot`.
    min_voxel_count : int, default 100
        Minimum number of voxels a region must have within the cluster to be reported.
    output_prefix : str, optional
        If provided, saves the significant clusters as MGH files 
        (e.g., {output_prefix}_lh.mgh).
    lh_target : ndarray, optional
        Left hemisphere target (replication) set statistical map (1D array of vertices).
        If provided, computes the replication rate of left hemisphere clusters in the target set.
    rh_target : ndarray, optional
        Right hemisphere target (replication) set statistical map (1D array of vertices).
        If provided, computes the replication rate of right hemisphere clusters in the target set.
    target_p_threshold : float, default 0.05
        P-value threshold for defining a voxel as "replicated" in the target/abcd set (used for replication rate calculation).


    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: hemi, cluster_size, area_mm2, cluster_p,
        peak_vertex, peak_stat.
    """
    # 1. Combined search-space geometry
    if fwhm is None:
        fwhm = np.nanmean([
            estimate_fwhm_surface(lh_stat, lh_mesh),
            estimate_fwhm_surface(rh_stat, rh_mesh),
        ])
    else:
        fwhm = float(fwhm)

    chi = lh_mesh["euler_char"] + rh_mesh["euler_char"]
    boundary = lh_mesh["boundary_length_mm"] + rh_mesh["boundary_length_mm"]
    area = lh_mesh["area_mm2"] + rh_mesh["area_mm2"]
    
    R = np.array([chi, boundary / (2.0 * fwhm), area / (fwhm * fwhm)])
    geometry = dict(R=R, area_mm2=area, fwhm=fwhm)

    # 2. Cluster search per hemisphere
    results = []
    sig_vertices = {"lh": [], "rh": []}
    for hemi, stat, mesh in [("lh", lh_stat, lh_mesh), ("rh", rh_stat, rh_mesh)]:
        stat = np.asarray(stat, float).squeeze()
        valid = mesh["mask"] & np.isfinite(stat)

        sets = []
        if sign in ["pos", "abs"]:
            sets.append((1, valid & (stat > cdt_z)))
        if sign in ["neg", "abs"]:
            sets.append((-1, valid & (stat < -cdt_z)))
        
        # Select data for current hemisphere
        annot = lh_annot if hemi == "lh" else rh_annot
        target_data = lh_target if hemi == "lh" else rh_target

        for s, active in sets:
            v = np.where(active)[0]
            if len(v) == 0:
                continue
                
            n, lab = connected_components(
                mesh["adjacency"][v][:, v], directed=False, return_labels=True
            )
            
            for i in range(n):
                cluster_v = v[lab == i]
                cluster_area = float(mesh["vertex_area"][cluster_v].sum())
                p_fwe, p_unc = _surface_cluster_p(cluster_area, cdt_z, geometry)

                if p_fwe < alpha:
                    # Keep track of significant vertices for saving
                    sig_vertices[hemi].extend(cluster_v)

                    vals = stat[cluster_v]
                    peak_idx = np.argmax(vals) if s > 0 else np.argmin(vals)
                    row = {
                        "hemi": hemi,
                        "cluster_size": len(cluster_v),
                        "area_mm2": cluster_area,
                        "cluster_p": p_fwe,
                        "peak_vertex": int(cluster_v[peak_idx]),
                        "peak_stat": float(vals[peak_idx]),
                    }

                    # Calculate overall replication rates (Optional)
                    if target_data is not None:
                        target_vals = target_data[cluster_v]
                        sig = (np.abs(target_vals) > stats.norm.ppf(1 - target_p_threshold/2)) & (np.sign(target_vals) == np.sign(vals))
                        row["target_rate"] = float(np.sum(sig) / len(cluster_v))

                    # Region annotation via external function
                    if annot is not None:
                        from ..atlas.regions import region_summary_freesurfer

                        row["brain_regions"] = region_summary_freesurfer(
                            cluster_vertices=cluster_v,
                            annot_data=annot,
                            discovery_img=stat,
                            target_img=target_data,
                            p_threshold=target_p_threshold,
                            min_voxel_count=min_voxel_count,
                        )

                    results.append(row)

        df = pd.DataFrame(results)

    # 3. Save MGH files if requested
    if output_prefix is not None:
        for hemi, stat in [("lh", lh_stat), ("rh", rh_stat)]:
            out_data = np.zeros_like(stat, dtype=np.float32)
            if len(sig_vertices[hemi]) > 0:
                valid_v = np.unique(sig_vertices[hemi])
                out_data[valid_v] = stat[valid_v]
                
                # Save as MGH (needs a 3D/4D shape, typically (N, 1, 1) for surfaces)
                if affine is None:
                    raise ValueError("`affine` is required to ensure correct spatial orientation for output files.")
                else:
                    save_affine = affine
                out_img = nib.MGHImage(out_data.reshape(-1, 1, 1).astype(np.float32), affine=save_affine)
                out_path = f"{output_prefix}_{hemi}.mgh"
                nib.save(out_img, out_path)
                print(f"Saved {hemi} significant clusters to {out_path}")
    
            else:
                print(f'Find no significant cluster!')

    return df