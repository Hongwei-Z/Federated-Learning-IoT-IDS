## Federated-Learning-IoT-IDS
### Federated Learning-based Anomaly and Intrusion Detection System for Smart Home Environments

### Author: Haoyu Wang & Hongwei Zhang


### File Description:    
1. Main Model: [Federated_IDS_Model.py](Federated_IDS_Model.py)

2. Dataset Preprocessing:
   1. CIC IoT-DIAD 2024 dataset   
      1. Multi-class classification: [DIAD-Multi.ipynb](DatasetPreprocessing/DIAD-Multi.ipynb)
      2. Binary classification: [DIAD-Binary.ipynb](DatasetPreprocessing/DIAD-Binary.ipynb)
   2. CIC IoMT dataset 2024   
      1. Multi-class classification: [IoMT-Multi.ipynb](DatasetPreprocessing/IoMT-Multi.ipynb)
      2. Binary classification: [IoMT-Binary.ipynb](DatasetPreprocessing/IoMT-Binary.ipynb)

3. Federated Learning Results:      
   1. Multi-class classification:     
      1. <img src="Results/Federated/multi/cnn/multi.csv_cnn_metrics_20250325_200259.png" width="70%">   
      2. <img src="Results/Federated/multi/mlp/multi.csv_mlp_metrics_20250325_200647.png" width="70%">   
      3. <img src="Results/Federated/multi/transformer/multi.csv_transformer_metrics_20250325_202147.png" width="70%">   
   2. Binary classification:   
      1. <img src="Results/Federated/binary/cnn/binary.csv_cnn_metrics_20250325_193756.png" width="70%">   
      2. <img src="Results/Federated/binary/mlp/binary.csv_mlp_metrics_20250325_194042.png" width="70%">   
      3. <img src="Results/Federated/binary/transformer/binary.csv_transformer_metrics_20250325_195043.png" width="70%">   

4. Centralized Learning Results:    
   1. [Multi-class classification](Results/Centralized/multi.csv)  
   2. [Binary classification](Results/Centralized/binary.csv)  


**Datasets:** 
1. [CIC IoT-DIAD 2024 dataset](https://www.unb.ca/cic/datasets/iot-diad-2024.html)    
2. [CIC IoMT dataset 2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

