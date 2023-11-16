from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
import h5py as h5

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil','attmil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping','kidney','camelyon16','camelyon16_again','resistance','cam16_masks','Mtb','Mtb2'])
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'kidney':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/kidney.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'g1':0, 'g2':1, 'g3':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'camelyon16':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon16.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'tumor':1, 'normal':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'camelyon16_again':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/camelyon16_again.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'tumor':1, 'normal':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'resistance':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/resistance_s1.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'g1':1, 'g0':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'cam16_masks':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/cam16_masks.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'tumor':1, 'normal':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'Mtb':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/Mtb.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'SS':0, 'C':1, 'A':2, "N":3},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'Mtb2':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/Mtb2.csv',
                            data_dir= os.path.join(args.data_root_dir),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'SS':0, 'C':1, 'A':2, "N":3},
                            patient_strat= False,
                            ignore=[])
else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        results = eval_heatmap(split_dataset, args, ckpt_paths[ckpt_idx])
        for i in results.keys():
            slide_id=results[i]['slide_id']
            A_raw=results[i]['A']
            label=results[i]['label']
            with h5.File(os.path.join(args.save_dir,str(slide_id)+'.h5'),'w') as f:
                f.create_dataset('A_raw',data=A_raw)
                f.create_dataset('label',data=label)
    print('Done')