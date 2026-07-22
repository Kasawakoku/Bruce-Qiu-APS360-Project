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

To be documented...  
```
python ./training/training.py 
```