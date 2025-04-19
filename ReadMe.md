# H-VIS 
Code of Paper 《High-Quality Neural Surface Reconstruction from Unoriented Point Clouds via Multilevel Tensor Product 
B-spline Hash Encoding and Viscosity Regularization》



![1](teaser/large_scan.png)
## Install
This code was tested with Python 3.10, torch 1.13.1 and cuda 11.7 on
a desktop PC equipped with an Intel i9-10980XE CPU (3.0 GHz), an NVIDIA GeForce RTX 3090 graphics card 
(24 GB memory) and Windows10 system.

Using conda to create the environment and activate it.
```
conda create -n H-VIS
conda activate H-VIS
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U kaleido
```
### Install pytorch3d (optional)
*Note that without this package, you can still run our experiment only if you don't add "--online_sampling" in 2d case 
or add "project" in "--second_order" (Of course, both of these are unrelated to the experiment in our paper)*

We use pytorch3d package for calculating the charmfer distance between two point clouds. Though it can also
be cauculated using scipy package, if you want to add it into your loss function or calculate it in per batch,
pytorch3d may be a better choice. Unfortunately, installing this package on Windows is a little complicated, the folowing
are two guides in English and Chinese.

1.https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

2.https://zhuanlan.zhihu.com/p/609391678

## Data
### Single model: 
we provide three 3D models and the points sample from their surfaces:
992113 (ABC), Armadillo (Famous), Screw (Real Scan). (*ps. the ground truth model of Screw is
dense point cloud.*)
### Dataset:
*ABC, Thingi10k, Famous* : follow the way provided in [point2surf](https://github.com/ErlerPhilipp/points2surf)

## Run
### Single model
```
python main_sdf.py 
--local_rank
0
--pts_path
data/Armadillo/Armadillo_pts.ply
--mesh_path
data/Armadillo/Armadillo_gt.ply
--workspace
SDF_armadillo
--encoding
B_grid
--optimizer
Adam
--lr_type
cos
--loss_func
l1
--minsurf
--second_order
vis
hessian
```
### Dataset

*Note*: The current code in ```test_on_dataset.py``` is designed specifically for the ABC, Thingi10k, and Famous datasets as 
listed in the provided link. To run the code with your own dataset, you need to make the following modifications:

*1.Ensure your dataset includes a testset.txt file:*

This file should list all the point cloud file names that you want to test. Each file name should be on a new line 
without extensions (e.g., cloud1, cloud2).

*2.Modify the ground truth file paths and extensions:*

Ensure that your dataset contains ground truth files (e.g., meshes for comparison). By default, the script assumes the 
ground truth files are located in a folder called ```03_meshes``` and have a ```.ply``` extension.
Update the folder name and file extension in lines 133 and 171 of ```test_on_dataset.py``` if your ground truth files 
are located in a different folder or have a different file extension.

Example: If your ground truth files are in a folder named ```ground_truth``` and have a ```.obj``` extension, modify the
script like this:
```
mesh_path = load_path + '/ground_truth/' + file + '.obj'
```

*3.Update the dataset name and directory in the command line:*

Replace the parameters ```$DATASET_NAME$``` and ```$DATASET_DIR$``` in the command line with the actual name and the local
path to your dataset. ```$DATASET_NAME$``` should be replaced with the name of your dataset folder (e.g., ```MyCustomDataset```).
```$DATASET_DIR$``` should be replaced with the path where your dataset is stored locally (e.g., ```/path/to/my/custom/dataset```).


Once you've made these changes, the script should be ready to run with your custom dataset.

```
python test_on_dataset.py 
--local_rank
0
--dataset_name
$DATASET_NAME$
--dataset_path
$DATASET_DIR$
--workspace
SDF_armadillo
--encoding
B_grid
--optimizer
Adam
--lr_type
cos
--loss_func
l1
--minsurf
--second_order
vis
hessian
```
## Contents
```
│  loss.py
│  main_sdf.py
│  ReadMe.md
│  setup.json
│  test_on_dataset.py
│  tpb_encoder.py
│  trainer.py
│  utils.py
│          
├─dataset
│  │  Dataset_1d.py
│  │  Dataset_2d.py
│  │  Dataset_3d.py
│  │  __init__.py
│          
├─network
│  │  network.py
│  │  SIREN.py
│  │  __init__.py
```
* *main_sdf.py*: run with a single model
* *test_on_dataset.py*: run with dataset
* *Dataset_1d.py*: create data of 1d shape(two points) for training.
* *Dataset_2d.py*: create data of 2d shape(circle, polygon...) for training.
* *Dataset_3d.py*: create data of 3d shape for training.
* *tpb_encoder.py*: tensor product B-spline encoder
* *network.py*: encoder + mlp
* *SIREN.py*: siren mlp
* *loss.py*: loss function
* *trainer.py*: training and evaluation
* *utils.py*: utility functions for visualization, evaluation and so on.
* *setup.json*: hyperparameters

## Setup.json：

### Basic Info
*seed*: random seed<br>
*input_dim*: length of input vector(1,2 or 3)<br>
*max_epochs*: the number of total epochs<br>
*init_lr*: the initial learning rate<br>
*num_samples*: the number of sampling points (only used for 1d, 2d and online_sampling 3d cases)<br>
*batch_size*: the number of points for per batch<br>
*batch_num*: the number of iterations for each epoch<br>
*iter*: the number of iterations for updating the parameters of network (you can set it larger than 1 and lower the batch_size 
when the memory is not enough)<br>
*grid_res*: resolution of Marching Cube algorithm<br>
### Loss
*coef_mnfld*: coefficient of the data term<br>
*coef_minsurf*: coefficient of the minimal surface term<br>
*coef_alpha*: parameter inside the minimal surface term<br>
*coef_eikonal*: coefficient of the eikonal and the viscosity term<br>
*coef_vis*: the initial coefficient of the laplacian trem inside the viscosity term<br>
*coef_div*: the initial coefficient of the hessian term<br>
*coef_div_down_type*: choose from "remain" (keep same), "digs" (down linearly and piecewisely) and "pow_full" (down exponentially)<br>
*coef_vis_down_type*: Same as above<br>
*digs_para*: the knot for "digs" down type<br>
*minsurf_mnfld*: whether contains points sampled from the surface when calculating
the minimal surface term<br>
*minsurf_near*: whether contains points sampled near the surface when calculating
the minimal surface term<br>
*minsurf_nonmnfld*: whether contains points sampled from the bounding box when calculating
the minimal surface term<br>
*E_contain_mnfld*: whether contains points sampled from the surface when calculating
the eikonal term<br>
*E_contain_near*: whether contains points sampled near the surface when calculating
the eikonal term<br>
*E_contain_nonmnfld*: whether contains points sampled from the bounding box when calculating
the eikonal term<br>
*div_contain_mnfld*: whether contains points sampled from the surface when calculating
the second order term (viscosity and hessian)<br>
*div_contain_near*: whether contains points sampled near the surface when calculating
the second order term<br>
*div_contain_nonmnfld*: whether contains points sampled from the bounding box when calculating
the second order term<br>

(*ps: In calculating the viscosity term, we utilized the intermediate results 
from the computation of the eikonal term, so if the "div_..." is set as true, 
the corresponding "E_..." should be set as true either.*)
### TPBEncoder
*grid_type*: choose from "hash" (using hash table), "dense" (not using hash table)<br>
*b-order*: B-spline degree + 1<br>
*num_levels*: levels of tensor product B-spline encoder<br>
*level_dim*: dimension of coefficient vector<br>
*base_resolution*: the number of B-spline basis of the first level - the degeree of B-spline (4 for 1d,2d and 16 for 3d)<br>
*desired_resolution*: the number of B-spline basis of the last level - the degeree of B-spline (256 for 1d, 2d and 1024 for 3d)<br>
*per_level_scale*: the resolution scaling factor between adjacent layers (This option will be disabled once the desired_resolution and num_levels are set, the defalut is 2)<br>
*log2_hashmap_size*: the result of taking the log base 2 of the maximum size of hash table<br>
*model_scale*: the radius of bounding box of model<br>
*space_scale*: the radius of bounding box of uniformed sampling points<br>
### MLP
*activation*: choose from "sine", "relu", "softplus", "tanh"<br>

# Acknowledgements
This code is referenced on [torch-ngp](https://github.com/ashawkey/torch-ngp), [POCO](https://github.com/valeoai/POCO) 
and [Neural-Singualr-Hessian](https://github.com/bearprin/Neural-Singular-Hessian).

Thanks to their impressive work.