# voxelgenetics


## BWAS (Brain-Wide Association Study) Pipeline

The `estimate_bwas` function runs GPU-accelerated Brain-Wide Association Studies on imaging phenotypes. It flexibly offers **three result-saving strategies**:
1. **`save_mode='h5'`**: Saves the complete chunked T-statistics into an HDF5 file. The data is organized by voxel chunks (e.g., `block_0_500`) with matrix shapes of `(n_snps, chunk_size)` (essential for downstream voxel-wise heritability or genetic correlation calculations).
2. **`save_mode='npy'`**: Filters and saves only the individual SNPs that show significant associations across a minimum number of voxels, storing them as lightweight `.npy` files (ideal for visualizing specific brain maps). You can control this filtering with `npy_t_threshold` (the absolute t-stat threshold, default 3.29) and `npy_min_sig_voxels` (the minimum number of significant voxels required to save a SNP, default 500).
3. **`save_mode='none'`**: Does not save any data to disk. Instead, it returns a Python dictionary containing the T-statistics directly in memory. The keys match the HDF5 block names (e.g., `block_0_500`) and the values are numpy arrays of shape `(n_snps, chunk_size)`. This is highly recommended if you plan to immediately calculate downstream metrics (like h2 or gencov) in the same script to avoid unnecessary disk I/O.

### File Loading and ID Matching

Before running BWAS, you need to prepare three main components: **Imaging data**, **Genotype data**, and **Covariates**. It is critical to ensure that all data matrices are strictly aligned by the **same set of individuals (Sample IDs)**. 

For a complete walkthrough on how to load data and perform analysis, please refer to the `generate_sim_data.ipynb` notebook included in the example folder. This notebook provides a quick-start guide to generating simulated data in a BIDS-compliant format, running GWAS and RFT analyses, and is ideal for testing the pipeline.

Once you have your data ready, here is a standard example of how to load and align the data before calling `estimate_bwas`:

#### Load Image Data
```python
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import nibabel as nib

# 1. Define Parallel Image Reader
def read_single_image(img_path, filetype, mask_bool=None, network_index=0, rh_img_path=None):
    """Reads a single image based on modality type."""
    if filetype == 'npy':
        # DU Network NPY
        data = np.load(img_path)[:, network_index].flatten()
    elif filetype == 'mgh':
        # FreeSurfer Surface (MGH) - Need to load both LH and RH
        lh_data = nib.load(img_path).get_fdata().flatten()
        rh_data = nib.load(rh_img_path).get_fdata().flatten()
        data = np.concatenate((lh_data, rh_data))
    elif filetype == 'nii.gz':
        # Standard NIfTI (e.g., VBM, DTI)
        data = nib.load(img_path).get_fdata()
        if mask_bool is not None:
            data = data[mask_bool]
        else:
            data = data.flatten()
    else:
        raise ValueError(f"Unsupported file type: '{filetype}'. Expected 'DU_*', 'mgh', or 'nii.gz'.")

    return data

# 2. Find and Load Imaging Data
image_dir = "data/imaging" 
modality = "VBM_2mm_s"   # Example: 'VBM_2mm_s', 'dti_FA_2mm', 'lh_thickness_fs5'
filetype = 'nii.gz'
mask_path = "data/MNI152_T1_2mm_brain.nii.gz"

# Load brain mask if applicable
mask_bool = nib.load(mask_path).get_fdata() > 0 if os.path.exists(mask_path) else None

# Find all subject folders (Assuming structure: data/imaging/{subject_id}/{modality}.nii.gz)
subject_folders = sorted(glob.glob(f"{image_dir}/*"))
imaging_ids = [int(os.path.basename(folder)) for folder in subject_folders]
image_paths = [os.path.join(folder, f"{modality}.nii.gz") for folder in subject_folders]
print(f"Loading {modality} data for {len(image_paths)} subjects in parallel...")

# Use joblib to read in parallel (prefer threads avoids pickle overhead)
results = Parallel(n_jobs=16, prefer="threads")(
    delayed(read_single_image)(img_path, filetype, mask_bool) 
    for img_path in image_paths
)

imaging_matrix = np.array(results)
imaging_ids = np.array(imaging_ids)
print(f"Loaded imaging matrix shape: {imaging_matrix.shape}")
```

#### Load Genotype Data
```python
from pandas_plink import read_plink1_bin

bed_file = "./example/genotypes/EUR.bed"
bim_file = "./example/genotypes/EUR.bim"
fam_file = "./example/genotypes/EUR.fam"

G = read_plink1_bin(bed_file, bim_file, fam_file, verbose=False)
genotype_ids = G.iid.values.astype(int)  # Extract Sample IDs from the PLINK file
plink_snps = G.snp.values                # Extract SNP IDs from the PLINK file
```
**❗ Important Note on Allele Direction**
`pandas-plink` typically codes genotypes as the count of the `a1` allele (where `0` = `a0/a0`), meaning the regression uses `a0` as the baseline and reports the effect of `a1`. In contrast, standard PLINK `--glm` uses `A2` as the baseline and reports the effect of `A1` (which corresponds to the effect of `a0`).

Because these two alleles are mutually exclusive, the resulting T-statistics from this package will have the same magnitude but the opposite sign compared to PLINK GWAS Z-scores. To align them, simply multiply the T-statistics by -1 or invert the genotype dosage matrix (`G = 2 - G`) prior to running the regression.

The exact allele counting behavior of pandas-plink can depend on its specific version. If you are unsure whether it counts `a1` or `a0`, you can verify it by calculating the Minor Allele Frequency (MAF) of a specific SNP and comparing it with public databases like [dbSNP (e.g., rs7287144)](https://www.ncbi.nlm.nih.gov/snp/?term=rs7287144):
- If `mean(G) / 2` matches the frequency of `a1`, the matrix counts `a1` (standard behavior).
- If `mean(G) / 2` matches the frequency of `a0`, the matrix counts a0.


#### ID matching
```python
import numpy as np

covariates = np.load("path/to/covariates.npy")
cov_ids = np.load("path/to/covariate_ids.npy")

common_ids = np.intersect1d(imaging_ids, genotype_ids)
common_ids = np.intersect1d(common_ids, cov_ids)

# Get the indices to align the matrices
_, idx_img, _ = np.intersect1d(common_ids, imaging_ids, return_indices=True)
_, idx_geno, _ = np.intersect1d(common_ids, genotype_ids, return_indices=True)
_, idx_cov, _ = np.intersect1d(common_ids, cov_ids, return_indices=True)

# Reorder the matrices to match the common IDs
aligned_imaging = imaging_matrix[idx_img, :]
aligned_genotype = 2 - np.array(G.values)[:, idx_geno]  # see Note on Allele Direction in data loading section
aligned_covariates = covariates[idx_cov, :]

print(f"Total matched samples: {len(common_ids)}")
```


### Basic Usage

Here is a standard call to run BWAS and save the raw t-statistics to an HDF5 file. To ensure the output HDF5 files are consistent for downstream analyses, pass the `sampleid` and `snplist` explicitly:

```python
import numpy as np
from voxelgenetics.bwas import estimate_bwas

# Run BWAS and save to HDF5
h5_file = estimate_bwas(
    genotype_bed=genotype_matrix,      # (n_samples, n_snps) numpy array
    imaging_matrix=imaging_data,       # (n_samples, n_voxels) numpy array
    sampleid=sampleid, 
    snplist=snplist,
    output_prefix="results/bwas_subset",
    
    # Covariate settings (See below for details)
    covariates=my_covariates,       # (n_samples, n_cov) 
    residualize_y=True,
    
    # Core parameters
    burden_threshold_ratio=0.01,  # Minimum MAF (minor allele frequency) to retain a burden
    chunk_size=500,               # Number of voxels processed per GPU batch
    burden_batch_size=100,        # Number of SNPs processed per GPU batch
    device='cuda:0',
    
    # Storage options
    save_mode='h5'  # 'h5', 'npy', or 'none'
)
```

If your goal is to visualize the brain maps of significant genetic associations, we recommend using `save_mode='npy'`. To reconstruct the 3D brain map, you need to pass the **complete brain imaging `imaging_matrix`** (used as input for BWAS) and the **specific SNP slice index** you want to calculate/visualize.

To convert these 1D `.npy` arrays back to 3D NIfTI images, you will need the **full brain mask** (e.g., `MNI152_T1_2mm_brain.nii.gz`) used during the imaging preprocessing.

```python
from voxelgenetics.bwas import estimate_bwas

# Run BWAS and save to HDF5
h5_file = estimate_bwas(
    genotype_bed=genotype_matrix,      # (n_samples, n_snps) numpy array
    imaging_matrix=imaging_data,       # (n_samples, n_voxels) numpy array
    sampleid=sampleid, 
    snplist=snplist,
    output_prefix="results/bwas_chr22",
    
    # Covariate settings (See below for details)
    covariates=my_covariates,       # (n_samples, n_cov) 
    residualize_y=True,
    
    # Core parameters
    burden_threshold_ratio=0.01,  # Minimum MAF (minor allele frequency) to retain a burden
    chunk_size=500,               # Number of voxels processed per GPU batch
    burden_batch_size=100,        # Number of SNPs processed per GPU batch
    device='cuda:0',
    
    # Storage options
    save_mode='npy',
    npy_t_threshold=3.29,
    npy_min_sig_voxels=500,
)
```

If you plan to perform **downstream genetic analyses (like voxel-wise heritability or genetic correlation)**, you need the complete T-statistics for all SNPs. The h5 or none modes are designed for this *(Please refer to the [Advanced Analysis Workflows](#advanced-analysis-workflows) section below for detailed implementations)*.


### Covariate Handling Strategies

Controlling for covariates (e.g., age, sex, head motion, genetic components) is crucial in imaging genetics. voxelgenetics provides three flexible ways to handle covariates during the BWAS pipeline, depending on your data status and computational preferences:

#### Strategy 1: Pre-regressed Imaging Data (Fastest)
If you have already regressed out the covariates from your imaging data using external software (e.g., FSL, SPM, or your own script) and saved the residualized matrix, you just need to tell the function how many degrees of freedom were consumed by those covariates to ensure accurate T-statistics.


```python
estimate_bwas(
    genotype_bed=genotype,
    imaging_matrix=pre_regressed_imaging, # Already residualized
    covariates=None,
    residualize_y=False,
    include_covariates_in_model=False,
    n_cov=51,  # Specify the number of covariates used (including intercept)
    ...
)
```

#### Strategy 2: Raw Imaging Data + Auto-Residualization (Recommended)
If you have raw imaging data and want the function to regress out the covariates before running the regression loop, set `residualize_y=True`. This performs the residualization in advance, significantly reducing GPU computational burden while maintaining mathematically equivalent results.
```python
estimate_bwas(
    genotype_bed=genotype,
    imaging_matrix=raw_imaging,
    output_prefix="results/bwas_residualized",
    
    covariates=my_covariates,      # Pass your covariate matrix
    residualize_y=True,            # Will regress out on CPU first
    include_covariates_in_model=False,
    ...
)
```

#### Strategy 3: In-Model Covariate Regression (Most Rigorous, GPU Intensive)
If you want to keep the covariates in the GPU design matrix during the regression, set `include_covariates_in_model=True`. This will **significantly increase GPU memory usage and computation time**, especially if the number of covariates is large.

```python
estimate_bwas(
    genotype_bed=genotype,
    imaging_matrix=raw_imaging,
    output_prefix="results/bwas_in_model",
    
    covariates=my_covariates,
    residualize_y=False,
    include_covariates_in_model=True, # Covariates go into the GPU design matrix
    ...
)
```


**❗ Important Note**: If `residualize_y=False` and `include_covariates_in_model=False` (we assume you have already regressed out covariates from the data), you must provide either `covariates` (to calculate n_cov) or explicitly set `n_cov`. The function will raise a ValueError otherwise to prevent incorrect degrees of freedom.

**💡 Tip:** You can actually set both `residualize_y=True` and `include_covariates_in_model=True` simultaneously. The imaging data will first be residualized in advance, and then the covariates will be included again in the GPU model. Overall, this has a negligible impact on the final statistical results, though it incurs both CPU preprocessing and GPU computational overheads.




## 3D Voxel-level RFT correction

Here is how to run cluster-extent RFT correction on a 3D statistical map.

```python
import nibabel as nib
from voxelgenetics.rft import cluster_correct_volume
from voxelgenetics.atlas import load_atlas_by_name

# 1. Load image and mask
img = nib.load("stat_map.nii.gz")
stat = img.get_fdata()
mask = nib.load("MNI152_T1_2mm_brain.nii.gz").get_fdata() > 0

# 2. (Optional) Load an atlas if you want brain region annotations
ukb_data, region_dict = load_atlas_by_name("vbm")

# 3. (Optional) Load a target replication map to calculate replication rates
target_img = nib.load("target_stat_map.nii.gz").get_fdata()

# 4. Run correction
df = cluster_correct_volume(
    stat_img=stat,
    mask=mask,
    cdt_z=3.29,
    alpha=5e-8,                # FWE significance threshold
    
    # Optional parameters:
    fwhm=None,                # None → automatically estimate FWHM from `stat_img`
                              # Or provide your own calculated value, e.g., fwhm=8.5 or (7, 7, 9.5)
    affine=img.affine,        # Provide → returns peak MNI coordinates in the output
    atlas=(ukb_data, region_dict), # Provide → returns brain region overlap in the output
    output_file="corrected_result.nii.gz", # Provide → saves a NIfTI file with only significant clusters
    min_voxel_count=100, # Use `min_voxel_count=100` to only report regions with ≥100 voxels in the cluster.

    target_img=target_img,    # Provide → returns replication rates (e.g., `target_rate` column) in the output
    target_p_threshold=0.05,  # P-value threshold for defining a voxel as "replicated" in the target/abcd set (default: 0.05)
)
```


**💡 Tip:** If you have raw 4D data and want to estimate the FWHM manually before running the correction, you can use `from voxelgenetics.rft import estimate_fwhm_3d`. Then pass the resulting scalar to the fwhm argument above.

```python
from voxelgenetics.rft.volume import estimate_fwhm_3d
fwhm_val = estimate_fwhm_3d(stat[..., None])[0] 
```

**💡 Tip:** We provide three built-in atlases in the [../data](https://github.com/qin1114/Voxel-genetics/tree/main/data) folder, including JHU for NODDI or DTI, Tian for T2* or QSM, and the UKB atlas for VBM (which parcellates the brain into 139 regions). Alternatively, you can load and use your own custom atlas.


## 2D Vertex-level Surface RFT Correction
For cortical surface data, the pipeline is similar but operates on left and right hemisphere meshes.
```python
import nibabel as nib
from voxelgenetics.rft.surface import (
    build_surface_mesh, 
    load_fs_annot, 
    cluster_correct_surface
)

# 1. Build surface meshes and load annotations
FSAVERAGE5 = "/path/to/fsaverage5"
lh_mesh = build_surface_mesh(f"{FSAVERAGE5}/surf/lh.white", f"{FSAVERAGE5}/label/lh.cortex.label")
rh_mesh = build_surface_mesh(f"{FSAVERAGE5}/surf/rh.white", f"{FSAVERAGE5}/label/rh.cortex.label")

lh_annot = load_fs_annot(f"{FSAVERAGE5}/label/lh.aparc.annot")
rh_annot = load_fs_annot(f"{FSAVERAGE5}/label/rh.aparc.annot")

ref_img = nib.load(f'{FSAVERAGE5}/STAT_FILE_fsaverage5_FWHM10mm.mgz')

# 2. Load statistical maps (Optional)
lh_stat = nib.load("lh_stat_file.mgh").get_fdata().squeeze()
rh_stat = nib.load("rh_stat_file.mgh").get_fdata().squeeze()

# Optional
from voxelgenetics.rft.surface import estimate_fwhm_surface
fwhm = np.nanmean([
    estimate_fwhm_surface(lh_stat, lh_mesh),
    estimate_fwhm_surface(rh_stat, rh_mesh),
])

# 3. (Optional) Load a target replication map to calculate replication rates
lh_target = nib.load("lh_target_file.mgh").get_fdata().squeeze()
rh_target = nib.load("rh_target_file.mgh").get_fdata().squeeze()

# 4. Run correction
df = cluster_correct_surface(
    lh_stat=lh_stat, 
    rh_stat=rh_stat,
    lh_mesh=lh_mesh, 
    rh_mesh=rh_mesh,
    cdt_z=3.29,
    alpha=5e-8,
    
    # Optional parameter:
    fwhm=fwhm, # None → automatically estimate FWHM from `stat_img`
                              # Or provide your own calculated value, e.g., fwhm=8.5

    lh_annot=lh_annot, # Provide → returns brain region overlap in the output
    rh_annot=rh_annot, 
    affine=ref_img.affine,
    output_prefix="corrected_surface", # Provide → saves significant clusters as `corrected_surface_lh.mgh` and `corrected_surface_rh.mgh`
    min_voxel_count=100, # Use `min_voxel_count=100` to only report regions with ≥100 voxels in the cluster.

    lh_target=lh_target,      # Provide → returns replication rates (e.g., `target_rate` column) in the output
    rh_target=rh_target,
    target_p_threshold=0.05,  # P-value threshold for defining a voxel as "replicated" in the target/abcd set (default: 0.05)
)
```



## Voxel-Wise Heritability (LDSC)

The `estimate_h2_voxel` function computes voxel-wise heritability (\(h^2\)) from the HDF5 files  or built-in data generated by the BWAS pipeline. It uses GPU-accelerated LD Score Regression (LDSC) to estimate the SNP-based heritability for each voxel across all chromosomes.

### Basic Usage

To calculate the heritability for every voxel, you need the chunked HDF5 files produced by `estimate_bwas` (with `save_mode='h5'`) and the corresponding LD score files.

```python
import numpy as np
from voxelgenetics.voxel.heritability import estimate_h2_voxel

# Define the chromosomes to process
chromosomes = list(range(1, 23))

# Path to LD score files (e.g., downloaded from Zenodo: https://zenodo.org/records/14993076/files/eur_w_ld_chr.tgz)
ld_folder = "/path/to/eur_w_ld_chr"

# Template for the HDF5 files. The `{chr}` placeholder will be replaced.
# Ensure your files are named like 'VBM_2mm_s_chr1.h5', 'VBM_2mm_s_chr2.h5', etc.
gwas_h5_template = "/path/to/results/VBM_2mm_s_chr{chr}.h5"

print("Running voxel-wise heritability...")
h2, h2_se = estimate_h2_voxel(
    chromosome_range=chromosomes,
    ld_score_folder=ld_folder,
    gwas_h5_template=gwas_h5_template,
    m_annot=1173569,     # Number of SNPs used for the reference panel (e.g., HM3)
    chunk_size=500,      # Number of voxels processed per GPU batch
    n_blocks=200,        # Number of jackknife blocks
    twostep=30,          # Two-step cutoff for LDSC
    device='cuda:0'      
)
```

**❗ Important Note**
1. The HDF5 files **must contain the complete SNP information for the selected voxels**. The heritability calculation performs LDSC regression across all SNPs in the reference panel; therefore, any missing SNPs (e.g., if they were filtered out during BWAS or saved using the `npy` mode) will result in inaccurate estimates or LDSC failures.
2. Voxel-wise LDSC involves intensive matrix operations. We highly recommend using a **GPU** (e.g., an NVIDIA A100 can compute 500 voxels for ~40,000 subjects in just a few seconds). Using `device='cpu'` is strongly discouraged for full-brain analyses, as it will be prohibitively slow. 


### Advanced Analysis Workflows

If you want to run BWAS and immediately calculate voxel-wise heritability **without the massive memory and disk overhead of writing HDF5 files**, use `save_mode='none'`. This mode returns a Python dictionary containing the T-statistics, SNP list, and sample IDs for each chromosome. You can pass this dictionary directly to `estimate_h2_voxel`.

Here is an example of looping through all chromosomes, storing results in memory, and running heritability:

```python
import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from voxelgenetics.bwas import estimate_bwas
from voxelgenetics.voxel import estimate_h2_voxel

chromosomes = list(range(1, 23))
ld_folder = "path/to/ld_score_folder"
data_save_dir = "path/to/plink_data"

bwas_dict = {}
# 1. Run BWAS for each chromosome (In-Memory)
for chr_num in chromosomes:
    print(f"Processing Chromosome {chr_num}...")
    
    # Load genotype for this chromosome
    bed_file = f'{data_save_dir}/UKB_gene_v3_imp_qc_chr{chr_num}_HM3.bed'
    bim_file = f'{data_save_dir}/UKB_gene_v3_imp_qc_chr{chr_num}_HM3.bim'
    fam_file = f'{data_save_dir}/UKB_gene_v3_imp_qc_chr{chr_num}_HM3.fam'
    
    fam_df = pd.read_csv(fam_file, sep='\s+', header=None, names=['family_id', 'individual_id', 'father_id', 'mother_id', 'sex', 'phenotype'])
    sampleid = fam_df['individual_id'].values
    
    bim_df = pd.read_csv(bim_file, sep='\t', header=None, names=['chromosome', 'variant_id', 'genetic_distance', 'position', 'allele1', 'allele2'])
    snplist = bim_df['variant_id'].values
    
    G = read_plink1_bin(bed_file, bim_file, fam_file, verbose=False)
    genotype_bed = 2 - np.array(G.values) # see Note on Allele Direction in data loading section

    # Run BWAS (returns a dict with 'blocks', 'snplist', 'sampleid', 'n_samples')
    bwas_dict[chr_num] = estimate_bwas(
        genotype_bed=genotype_bed,
        imaging_matrix=imaging_matrix,
        covariates=covariates,     
        residualize_y=True,  
        include_covariates_in_model=False,
        sampleid=sampleid, 
        snplist=snplist,
        burden_threshold_ratio=0.01,
        chunk_size=500,
        burden_batch_size=50,
        device='cuda:0',
        save_mode='none', # Return dict in memory
    )

# 2. Run Voxel-wise Heritability directly from the dictionary
print("Running voxel-wise heritability...")
h2, h2_se = estimate_h2_voxel(
    chromosome_range=chromosomes,
    ld_score_folder=ld_folder,
    bwas_dict=bwas_dict,       # Pass the in-memory dict here
    gwas_h5_template=None,     # Explicitly set to None
    chunk_size=500,            # Matched with chunk_size used for bwas estimation
    n_blocks=200,
    device='cuda:0'
)
```


## Voxel-Wise Genetic Correlation (LDSC)

The `estimate_rg_voxel` function computes the voxel-wise genetic correlation (`rg`) between brain voxels and a target phenotype (e.g., a disease or a complex trait). 

### Basic Usage

To calculate genetic correlation, you need:
1. The BWAS results (either HDF5 files from `save_mode='h5'` or an in-memory dictionary from `save_mode='none'`).
2. The target phenotype GWAS summary statistics file. Specify the column names in your target GWAS file with `target_snp_col, target_z_col, target_n_col`

```python
import numpy as np
from voxelgenetics.voxel import estimate_rg_voxel

chromosomes = list(range(1, 23))
ld_folder = "path/to/ld_score_folder"

rg, rg_se, gencov, gencov_se = estimate_rg_voxel(
    chromosome_range=chromosomes,
    ld_score_folder=ld_folder,
    
    # Provide the path to your target phenotype GWAS file
    target_gwas_path='path/to/Target_Trait_GWAS.txt', 
    
    # Provide either the HDF5 template or the in-memory BWAS dict
    gwas_h5_template='path/to/results/bwas_chr{chr}.h5',
    # bwas_dict=my_in_memory_dict, 
    
    chunk_size=200,
    n_blocks=200,
    device='cuda:0',
    
    # Specify the column names in your target GWAS file
    target_snp_col='SNP',   # e.g., 'SNP', 'ID', 'rsid'
    target_z_col='Z',       # e.g., 'Z', 'Z_STAT'
    target_n_col='N'        # e.g., 'N', 'OBS_CT'
)
```

**💡 Tip**: For more details on how to use the h5 file template or the in-memory bwas_dict, *please refer to the [Voxel-Wise Heritability](#advanced-analysis-workflows)* section above. The data loading logic is identical.
