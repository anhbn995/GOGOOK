# RoadNetKeras
1. Download data :
	- The dataset can be downloaded from https://github.com/yhlleo/RoadNet

2. File annotation :
	- data_loader.py : Contains functionalities to load training and testing data from the data folder.
	- models.py : Defines the architectures of the sub-networks.
	- roadnet.py : Defines the complete architecture of roadnet and loss customization.
	- train.py : The main training scripts.

3. Training :
	- Run the following command (Linux) : ```console $ python3 train.py ```
	- Resetting some parameters :
	$ python3 train.py -d <data_folder>
						-n <number_of_training_images>
						-i <number_of_epochs>
						-p <number_of_patient_epochs>
						-c <checkpoint_path>
