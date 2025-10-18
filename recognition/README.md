# Recognition Tasks

This workspace collects my COMP3710 pattern-recognition projects. Each subfolder is developed in phases so the commit history captures meaningful milestones rather than a single monolithic drop.

## Completed so far
- ✅ **3D Improved U-Net (Hard difficulty)**  
  - Set up prostate MRI dataset loader with percentile-based normalisation and patch sampling (`3D_Improved_UNet-48593577/dataset.py`).  
  - Documented the training recipe, dataset details, and reproducibility steps in the project report README.  
  - Confirmed data loading pipeline by running the training bootstrap script (`train.py`) with sample batch logging.

## In progress / Upcoming
- 🔄 Integrating the attention-augmented 3D U-Net modules and hybrid Dice+Focal loss in `modules.py`.  
- ⏳ Full training sweep and quantitative evaluation once the model architecture is finalised.  
- 📈 Aggregate results and qualitative visualisations for inclusion in the final report.

Feel free to explore individual project folders for more detailed notes, configs, and scripts as each phase lands. Every major feature will arrive with its own commit to keep the implementation trail clear.*** End Patch
