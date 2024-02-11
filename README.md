#Exploring the Depths of 3D Semantic Novelty Detection

##The original reference code from
SemNov_AML_DAAI_23-24: https://github.com/antoalli/SemNov_AML_DAAI_23-24


## Introduction

This code allows to replicate all the experiments and reproduce all the results that we included in our project report. 

### Requirements
We perform our experiments on the Google Colab environment.
For the 3DOS bechmark and the faliure cases analyses:

```bash
!pip install timm==0.5.4 wandb tqdm h5py==3.6.0 protobuf==3.20.1 lmdb==1.2.1 msgpack-numpy==0.4.7.1 ninja==1.10.2.2 scikit-learn
!pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```
For the OpenShape test:
```bash
!pip install timm==0.5.4 wandb tqdm h5py==3.6.0 protobuf==3.20.1 lmdb==1.2.1 msgpack-numpy==0.4.7.1 ninja==1.10.2.2 scikit-learn
!pip install huggingface.hub wandb torch_redstone numpy dgl einops utils torchlars
!pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```
**Additional libraries** 

N.B. to install PointNet++ ops the system-wide CUDA version must match the PyTorch one (CUDA 11 in this case).

```bash
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"
```

### Data
Use the prepared script to download all datasets. 

```bash
chmod +x download_data.sh
./download_data.sh
```

A common root for all datasets will be created in project dir, by default named *3D_OS_release_data* .
```
3D_OS_release_data (root)
├─ ModelNet40_corrupted
├─ sncore_fps_4096
├─ ScanObjectNN
├─ modelnet40_normal_resampled
```

The absolute path to the datasets root must be passed as **--data_root** argument in all scripts.


## Run example

## RUNNING ON COLAB FOR STUDENTS
<span style="color:red">
In this repository we use PyTorch Distributed Training that assumes an environment/machine with multiple GPUs. To run experiments on Google Colab (single GPU machine) you need to remove "-m torch.distributed.launch --nproc_per_node=1" from the bash commands in the guidelines.
</span>

### Example
```bash
# Original bash command from README with PyTorch DDP:
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 

# [AML/DAAI STUDENT] What you should run on COLAB or single GPU environment:
python classifiers/trainer_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 
```
The same also holds for the eval commands. 


### Training

Each experiment requires to choose a backbone (through the config file), a loss function and a source set. For example: 

```bash
# multiple gpus
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE 
```

Details: 

 - the training seed can be specified via `--seed <int>` argument. Seeds used for our paper experiments
   are: 1, 41, 13718. 
 - the source set determines which class set will be used for *known* classes in the given experiment. In case of synth->synth experiments the other two sets 
   among [SN1, SN2, SN3] will define *unknown* classes. In case of synth->real experiments the
   *unknown* classes set is composed of the other among [SR1, SR2] and a common OOD set. In practice
   in each experiments there are two different sets of unknown classes.
 - training output is stored in `outputs/<exp_name>`. 

### Eval 

```bash
# multiple gpus
python -m torch.distributed.launch --nproc_per_node=1 classifiers/trainer_cla.py --config cfgs/dgcnn-cla.yaml --exp_name DGCNN_CE_SN1 --src SN1 --loss CE -mode eval --ckpt_path outputs/DGCNN_CE_SN1/models/model_last.pth

```

Example output:
```bash
Computing OOD metrics with MLS normality score...
AUROC - Src label: 1, Tar label: 0
Src Test - Clf Acc: 0.8522321428571429, Clf Bal Acc: 0.7607892805466309
Auroc 1: 0.7345, FPR 1: 0.7905
Auroc 2: 0.7574, FPR 2: 0.7458
Auroc 3: 0.7440, FPR 3: 0.7719
to spreadsheet: 0.734540224202969,0.7904599659284497,0.7573800391095067,0.7457882069795427,0.7440065016031351,0.7719451371571072
```

The output contains closed set accuracy on known classes (Clf Acc), balanced closed set accuracy
(Clf Bal Acc) and 3 sets of open set performance results. In the paper we report AUROC 3 and FPR 3
which refer to the scenario `(known) src -> unknown set 1 + unknown set 2`.
