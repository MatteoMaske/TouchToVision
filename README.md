# TouchToVision
TouchToVision project for the Computer Vision course 2023/24

## Description
The project is about the development of a system which able to pair a visual frame with a tactile frame. The dataset used for the project is a subset of the original [TouchAndGo](https://github.com/fredfyyang/Touch-and-Go) dataset. The proposed pipeline is able to extract from the visual video and tactile video the most relevant frames and then pair them. 

## Dataset
The dataset is composed of 4 objects sampled from the TouchAndGo dataset. The processed object are saved in the `dataset` folder. The dataset is composed of 4 objects: `rock`, `rock2`, `tree`, `grass`. Each object is composed divided in 2 folders, one for the visual frames and one for the tactile frames

## Repository structure
The repository is structured as follows:
- `data`: folder containing the dataset
- `src`: folder containing scripts used to test monocular depth estimation models
- `weights`: folder containing the weights of the super resolution model [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- `sampler.py`: script used to sample the visual and tactile frames from the dataset
- `matcher.py`: script used to match the visual and tactile frames once they have been sampled using the sampler

## Usage
First, create a virtual environment and install the required packages:
```bash
pip install -r requirements.txt
```

Then, run the sampler script to sample other frames from the dataset (you can already find the sampled frames in the `data` folder):
```bash
python sampler.py -i rock 
```

Finally, run the matcher script to match the visual and tactile frames among different objects:
```bash
python matcher.py --class rock
```

## Authors
- [Davide Cerpelloni](https://github.com/davidecerpelloni)
- [Matteo Mascherin](https://github.com/MatteoMaske)