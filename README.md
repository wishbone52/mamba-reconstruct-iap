## PIAP-Mamba


### 🚀 Installation
Clone the repository and set up the Conda environment:
```bash
#!/bin/bash
# Create conda environment
conda env create -f environment.yml
conda activate mamba_env
```

### 🧪 Download data
Run the downloader script to fetch the raw iAPs/eAPs dataset:
```bash
# download the raw data
cd src/data
python data_downloader.py
```

### 🔨 How to Run
The training process comprises two stages: teacher model training, followed by the distillation-based training of `piap_mamba` which simultaneously incorporates physics constraints.

The `src/piap_trainer.py` script provides a unified Trainer class that supports both training phases. To specify the desired training stage (or to run evaluation), configure the corresponding arguments in `configs/training_config.yaml`.

```bash
python scrips/train.py --config 'configs/training_config.yaml'
```

### 📊 Results Visualization
For results visualization, after evaluation, you can simply call the functions in `src/plotters.py`, such as `plot_samples()` to visualize the results.
