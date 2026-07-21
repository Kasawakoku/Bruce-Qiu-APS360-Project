# Bruce-Qiu-APS360-Project
Repository for the **Ground-Level Fine-Grained Aircraft Variant and
Airline Classification via Multi-Task Deep Learning** Project by Bruce Qiu, done for the APS360 Introducation to Deep Learning at the University of Toronto.  
The main training script and CSVs recording training metrics for each model are stored in the training folder.  
The data folder consists of images (untracked), class definitions to define airline and aircraft hierarchy mapping, metadata CSVs that define metadata such as photo ID, aircraft variant, airline, and photographer attribution for each image, as well as data processing scripts.   
For information about the 2013 FGVC-Aircraft Benchmark by Maji et al., which is used by this project as a fallback dataset as well as reference for the aircraft hierarchy definitions in aircraft_hierarchy.yaml, please visit https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/.  
  

To clone repository:  

```
git clone https://github.com/Kasawakoku/Bruce-Qiu-APS360-Project
cd Bruce-Qiu-APS360-Project
```

Set up the Python environment as desired. Then install dependencies: 

```
pip install -r requirements.txt
```