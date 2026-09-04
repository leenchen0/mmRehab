
# SMPL model files

SMPL model files are intentionally excluded because they are distributed under a separate license.

1. Register at the [official SMPL website](https://smpl.is.tue.mpg.de/) and accept its model license.
2. Download the Python 2.7 model archive and place these files in this directory:
   - `basicModel_f_lbs_10_207_0_v1.0.0.pkl`
   - `basicmodel_m_lbs_10_207_0_v1.0.0.pkl`
3. From `code/smpl`, run `python extract_SMPL_model.py ./models`.

The conversion creates `code/smpl/smpl_f.pkl` and `code/smpl/smpl_m.pkl`. These generated files are also ignored by Git.
