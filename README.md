# Bruce-Qiu-APS360-Project
## Introduction
This is the repository for the **Ground-Level Fine-Grained Aircraft Variant and Airline Classification via Multi-Task Deep Learning** Project by Bruce Qiu, done for the APS360 Introducation to Deep Learning at the University of Toronto.  
The main training and evaluating scripts, as well as CSVs recording training metrics for each model are stored in the `training` folder.  
The `Data` folder consists of images (untracked), class definitions to define airline and aircraft hierarchy mapping, metadata CSVs that define metadata such as photo ID, aircraft variant, airline, and photographer attribution for each image, as well as data processing scripts. For explicit photographer and copyright attribution for each image, please check `./Data/metadata/airliners_metadata.csv`.  

## Some Resources Used   
[Airliners.net](https://www.airliners.net/) and its [photo database](https://www.airliners.net/search) is the main photo database used by the project. Its [terms of use](https://www.verticalscope.com/aboutus/tos.php?site=airliners.net) grants use of the site to "download any content made available on the Web Site" for "non-commercial, personal, or educational purposes". 
[The 2013 FGVC-Aircraft Benchmark](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) by Maji et al. is used by this project as a fallback dataset as well as reference for the aircraft hierarchy definitions in `./Data/class_definitions/aircraft_hierarchy.yaml`.  
  
## Setup
To clone repository:  

```
git clone https://github.com/Kasawakoku/Bruce-Qiu-APS360-Project
cd Bruce-Qiu-APS360-Project
```

Set up the Python environment as desired. Then install dependencies: 

```
pip install -r requirements.txt
```

## Data Processing Pipeline
  
Follow these steps for the complete data processing pipeline.  
  
To scrape images and associated metadata:

```
python ./Data/scripts/scraping.py [start_page] [end_page] 
```

To update list of all airlines and aircraft models are in each from the metadata:  

```
python ./Data/scripts/distribution_reports.py
```

To update list of all manufacturers, families, variants, models, airlines, and how many images are in each from the metadata, after trimming out airline and variant classes with low number of images (default threshold is 90):  

``` 
python ./Data/scripts/hierarchical_reports_trimmed.py [airline_threshold] [variant_threshold]
```

To output metadata with invalid airlines trimmed, with invalid variants trimmed as well as the AND, OR and XOR of these sets, after referencing the airlines/aircraft mapping and trimming out airline and variant classes with low number of images (default threshold is 90):  

``` 
python ./Data/scripts/update_metadata_trimmed.py [airline_threshold] [variant_threshold]
```

To split the metadata into training, validation, and testing sets with stratified splitting (default 80-10-10 split), with strata below a threshold (default 5) being sorted into a "rare" stratum for spltiting:

``` 
python ./Data/scripts/split_csvs_new.py [validation_fraction] [testing_fraction] [threshold]
```

To check the splitting for the classes most skewed towards training/validation/testing, as well as output splitting graphs either as a whole or with class filters (use `"None"` if filter not desired for either positions. The progress report uses `airline_filter="OTHERS"` and `airline_filter="Austrian Airlines"`):

```
python ./Data/scripts/check_splits.py [airline_filter] [variant_filter] 
```

## Training and Evaluation Pipeline

The neural network training, evaluation, and prediction pipeline is consolidated into a single entry point (`main.py`) which can be controlled entirely via command-line arguments. This makes it ideal for submitting remote SLURM jobs.

### Available Command-Line Arguments

**Core Execution:**
* `--mode`: The execution mode. Choices: `train`, `sanity`, `predict`, `test` *(Required)*.
* `--model`: The architecture to use. Choices: `baseline_variant`, `primary` *(Required)*.
* `--run_name`: Optional custom name for the run (useful for ablation studies). Overrides default naming.

**Hyperparameters:**
* `--batch_size`: Batch size for DataLoaders (Default: `16`).
* `--lr`: Learning rate for the Adam optimizer (Default: `0.001`).
* `--epochs`: Total number of epochs to train for (Default: `30`).
* `--image_size`: Dimensions to resize the square-padded images to (Default: `300`).
* `--num_workers`: Number of background CPU processes for data loading. Ensure your SLURM script requests at least this many CPU cores (Default: `4`).

**Checkpointing & Logging:**
* `--checkpoint_dir`: Directory to save/load `.pt` model files (Default: `checkpoints`).
* `--resume_checkpoint`: File path to a specific `.pt` file to resume training or run inference from.
* `--checkpoint_freq`: Save a model checkpoint every X epochs (Default: `1`).
* `--record_freq`: Record iteration-level loss/F1 metrics every X batches (Default: `100`).

**Data Paths:**
* `--data_dir`, `--train_csv`, `--val_csv`, `--test_csv`, `--airline_csv`, `--variant_csv`: Use these to override default relative data paths if running from a different directory.
* `--image_path`: Path to a single image file *(Required ONLY when `--mode predict` is used)*.

---

### Sample Commands

#### 1. Training from Scratch (Baseline Model)
Trains the baseline variant model for 35 epochs and saves checkpoints to a dedicated folder.
```
python ./training/main.py --mode train --model baseline_variant --epochs 35 --lr 0.001 --batch_size 16 --checkpoint_dir "checkpoints_variant_baseline"
```

#### 2. Resuming Training (Primary Model)
Resumes training the multi-task primary model starting from a previously saved epoch. The script automatically reads the last completed epoch from the `.pt` file and continues up to the specified `--epochs` target.
```
python ./training/main.py --mode train --model primary --epochs 10 --checkpoint_dir "checkpoints_primary" --resume_checkpoint [checkpoint_directory]
```

#### 3. Running an Ablation Study (Custom Naming)
Use the `--run_name` flag to easily tag experimental variations of your models. Checkpoints and CSV logs will inherit this name.
```
python ./training/main.py --mode train --model baseline_variant --epochs 20 --run_name "Ablation_NoDropout_Test"
```

#### 4. Sanity Check (Small Dataset Overfit)
Runs the training loop on an extremely small subset of data (16 images) to ensure the model can overfit and that backpropagation is working correctly without crashing.
```
python ./training/main.py --mode sanity --model baseline_variant --epochs 30 --batch_size 8 --checkpoint_dir "checkpoints_sanity"
```

#### 5. Manual Evaluation (Single Image Prediction)
Loads a single image from disk, applies padding and standard transforms, and outputs the human-readable class predictions. *(Note: The primary model will automatically output both Variant and Airline predictions).*
```
python ./training/main.py --mode predict --model primary --image_path [image_directory] --resume_checkpoint [checkpoint_directory]
```

#### 6. Final Evaluation (Test Set)
Runs the best saved checkpoint exclusively on the unseen `test_csv` split to calculate final Loss and Weighted F1-Scores.
```
python ./training/main.py --mode test --model primary --resume_checkpoint [checkpoint_directory]
```

#### 7. Generate Training Graphs (From Checkpoint)
Reads a model's `.pt` checkpoint to automatically extract the hyperparameters, locate the associated CSV logs, and save the resulting graphs locally as PNG images.
```bash
python ./training/main.py --mode graph --model primary --checkpoint_dir "checkpoints_primary" --resume_checkpoint [checkpoint_directory]
```