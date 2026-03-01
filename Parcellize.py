# parcellize.py
import numpy as np
from glob import glob
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
import os

# Atlas
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
masker = NiftiLabelsMasker(
    labels_img=schaefer.maps,
    standardize=True,
    detrend=True,
    t_r=2.0,
    high_pass=0.01,
    low_pass=0.1
)

os.makedirs('data/timeseries', exist_ok=True)

nii_files = sorted(glob(
    'data/derivatives/fmriprep/sub-*/func/*task-rest*MNI152NLin2009cAsym_preproc.nii.gz'
))

print(f"{len(nii_files)} sujets trouvés")

for nii_path in nii_files:
    subject_id = nii_path.split('/')[3]  # sub-10159
    out_path = f'data/timeseries/{subject_id}.npy'
    
    if os.path.exists(out_path):
        print(f"  skip {subject_id} (déjà fait)")
        continue
    
    try:
        ts = masker.fit_transform(nii_path)  # [T, 400]
        ts = ts.T                             # [400, T]
        np.save(out_path, ts)
        print(f"  ✅ {subject_id} : {ts.shape}")
    except Exception as e:
        print(f"  ❌ {subject_id} : {e}")

print("Parcellisation terminée")