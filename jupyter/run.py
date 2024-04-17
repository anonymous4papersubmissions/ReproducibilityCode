import yaml
import os
import pandas as pd
import numpy as np
from pprint import pprint

from experiment import PruningExperiment
from experiment.visualizations import VisualizationExperiment
from experiment.consistency import ConsistencyExperiment

DELIMITER = '_-_'

PATH_ROOT = None
os.environ['DATAPATH'] = None

EXT_LI = ['run0', 'run1', 'run2']
EXT = 'run0'

do_train = True
do_finetune = True 
do_visualization = False 
do_consistency = False 
do_print_results = False

config_dataset_pn_li = [
    #'OpenML_detailed.yaml',
    #'OpenML_pt1_more_23.yaml',
    #'OpenML_detailed.yaml',
    #'OpenML_pt1_more.yaml',
    'OpenML.yaml',
    # 'OpenML_pt2.yaml',
    #'OpenML_pt1_more_06.yaml',
    #'OpenML_pt1_more_23.yaml',
    #'OpenML_pt2_more.yaml'
    #'OpenML_pt2_remaining.yaml'
    #'OpenML_pt1_more_06.yaml'
    #'OpenML_detailed.yaml',
    #'OpenML_pt1_more_23.yaml'
    #'CIFAR10.yaml',
    #'CIFAR100.yaml',
    #'SVHN.yaml'
]
config_model_pn_li = [
    #'resnet.yaml'
    'MLP_TAB.yaml'
    #'MLP_CV.yaml'
]
config_method_pn_li = [
    # 'SUP.yaml',
    'SUP_SCL_pi_0_8_TAB.yaml',
    #'SUP_SCL_pi_0_125_TAB.yaml', 
    
    #'SUP.yaml',
    #'SUP_SCL_pi_0_125_CV.yaml',
    #'SUP_SCL_pi_0_0625_CV.yaml',

    #'SUP_SCL_pi_0_125_TAB.yaml',
    #'SUP_SCL_pi_0_125_CV.yaml',
    #'SUP_SCL_pi_0_0625_CV.yaml',
    #'betalasso2e-5.yaml'

    #'SUP_pi_0_5.yaml', 
    #'SUP_SCL_TAB.yaml', 
    #'SUP_pi_0_25.yaml', 
    #'SUP_pi_0_0625.yaml',
    #'SCL_TAB.yaml', 
    #'SUP_SCL_TAB.yaml', 

    #'SUP_SCL_pi_0_125_TAB.yaml', 
    #'SUP_pi_0_25.yaml', 
    #'SUP_SCL_TAB.yaml', 
    #'SUP_SCL_CV.yaml', 
    #'SUP_SCL_pi_0_125_CV.yaml', 
    #'SCL_CV.yaml',
    #'SCL_CV.yaml', 'SUP_pi_0_125.yaml', 'SUP_SCL_pi_0_125_CV.yaml',
    #'SUP_pi_0_5.yaml', 'SUP_pi_0_25.yaml', 'SUP_pi_0_0625.yaml',
    #'SUP_SCL_pi_0_0625_CV.yaml',
    #'SUP_SCL_pi_0_125_CV.yaml',
    #'betalasso1e-6.yaml', 'betalasso2e-6.yaml', 'betalasso5e-6.yaml', 'betalasso1e-5.yaml', 'betalasso2e-5.yaml'
]


'''
config_dataset_pn_li = [
    'CIFAR10.yaml',
    'CIFAR100.yaml',
    'SVHN.yaml'
]
config_model_pn_li = [
    'MLP_CV.yaml'
]
'''
'''
config_dataset_pn_li = [
    'CIFAR10.yaml',
    'CIFAR100.yaml',
    'SVHN.yaml'
]
config_model_pn_li = [
    'MLP_CV.yaml'
]
config_method_pn_li = [
    'SUP.yaml', 'SUP_pi_0_125.yaml', 'SUP_SCL_CV.yaml'
] # 2(3) + 8
config_method_pn_li = [
    'SUP_pi_0_5.yaml', 'SUP_pi_0_25.yaml', 'SUP_pi_0_0625.yaml'
] # 3 + 12
'''

# TODO: what is best pi? -> run SUP + SCL + pi

# config_method_pn_li = [
#     'SUP.yaml', 'SUP_pi_0_125.yaml', 'SUP_SCL_CV.yaml', 'SUP_SCL_pi_0_125_CV.yaml'
# ]
'''
config_dataset_pn_li = [
    'OpenML_detailed.yaml'
]
config_model_pn_li = [
    'MLP_TAB.yaml'
]
config_method_pn_li = [
    'betalasso1e-6.yaml', 'betalasso2e-6.yaml', 'betalasso5e-6.yaml',
    'betalasso1e-5.yaml', 'betalasso2e-5.yaml'
] # 2
config_dataset_pn_li = [
    'OpenML_detailed.yaml'
]
config_model_pn_li = [
    'MLP_TAB.yaml'
]
config_method_pn_li = [
    'SUP.yaml', 'SUP_pi.yaml', 'SUP_SCL_TAB.yaml', 'SUP_SCL_pi_TAB.yaml'
]
 
config_method_pn_li = [
    'betalasso1e-6.yaml', 'betalasso2e-6.yaml', 'betalasso5e-6.yaml',
    'betalasso1e-5.yaml', 'betalasso2e-5.yaml'
]
'''

config_pn_li = []
for config_dataset_pn in config_dataset_pn_li:
    for config_model_pn in config_model_pn_li:
        for config_method_pn in config_method_pn_li:
            config_pn_li.append((
                os.path.join(PATH_ROOT, 'dataset', config_dataset_pn),
                os.path.join(PATH_ROOT, 'method', config_method_pn),
                os.path.join(PATH_ROOT, 'model', config_model_pn)))

def main():
    ext = EXT
    acc_dict = {}
    print_results_out = []
    for config_dataset_pn, config_method_pn, config_model_pn in config_pn_li:
        with open(config_dataset_pn, 'r') as fp:
            config_dataset = yaml.safe_load(fp)
        with open(config_method_pn, 'r') as fp:
            config_method = yaml.safe_load(fp)
        with open(config_model_pn, 'r') as fp:
            config_model = yaml.safe_load(fp)

        config = {}
        config.update(config_method)
        config.update(config_dataset) # can overwrite 'small_init_multiplier' hyperparameter
        config.update(config_model) # can overwrite 'save_freq'
        #if ext != 'run0':
        config['save_freq'] = None

        for dataset_name in config['dataset_li']:
            if do_train:
                print(f'config:\n{config_dataset_pn}\n{config_method_pn}\n{config_model_pn}')
                pprint(config)
                run_train(config, dataset_name, ext=ext)
            if do_finetune:
                print(f'config:\n{config_dataset_pn}\n{config_method_pn}\n{config_model_pn}')
                pprint(config)
                run_finetune(config, dataset_name, ext=ext)
            if do_visualization:
                ext = 'run0'
                run_visualization(config, dataset_name, ext=ext)
            if do_consistency:
                if 'SUP.yaml' in config_method_pn or 'beta_lasso' in config_method_pn:
                    compression_li = [1]
                else:
                    compression_li = [8] if 'OpenML' in dataset_name else [16]
                run_consistency(config, dataset_name, compression_li, ext=ext)
            if do_print_results:
                dn = \
                    f"{dataset_name}{DELIMITER}{config['model']}{DELIMITER}{config['small_init_multiplier']}" + \
                    f"{DELIMITER}{config['contrastive']}{DELIMITER}{config['beta_lasso']}{DELIMITER}"
                acc_dict[dn] = get_results(config, dataset_name, ext_li=EXT_LI)
                if config['beta_lasso'] is None:
                    #for compression in [2,4,8,16]:
                    for compression in ([8] if 'OpenML' in dataset_name else [2,4,8,16]):
                        acc_dict[dn + f'{compression}'] = \
                            get_results(
                                config, dataset_name, ext_li=[
                                    f'{e}{DELIMITER}{compression}{DELIMITER}ft'
                                    for e in EXT_LI])

    print_results(acc_dict)

def get_logdir(config, dataset_name, ext=''):
    #os.system(f"mkdir {os.path.join(PATH_ROOT, 'logs')}")
    dn = \
        f"{dataset_name}{DELIMITER}{config['model']}{DELIMITER}{config['small_init_multiplier']}{DELIMITER}" + \
        f"{config['contrastive']}{DELIMITER}{config['beta_lasso']}{DELIMITER}{ext}"
    pn = os.path.join(PATH_ROOT, 'logs', dn)
    return pn

def print_results(acc_dict):
    for dn, acc in sorted(acc_dict.items()):
        print(
            f"{dn.replace(DELIMITER,',')},{','.join(acc['valid_acc1'])}" + \
            f",{','.join(acc['valid_acc5'])},{','.join(acc['test_acc1'])}" + \
            f",{','.join(acc['test_acc5'])}")

def get_results(config, dataset_name, ext_li):
    valid_acc1_li, valid_acc5_li, test_acc1_li, test_acc5_li = [], [], [], []
    for ext in ext_li:
        try:
            valid_acc1, valid_acc5, test_acc1, test_acc5 = \
                get_results_single_logdir(config, dataset_name, ext=ext)
            valid_acc1_li.append(valid_acc1)
            valid_acc5_li.append(valid_acc5)
            test_acc1_li.append(test_acc1)
            test_acc5_li.append(test_acc5)
        except:
            print(f'missing directory: {dataset_name, config, ext}')
    acc_dict = {
        'valid_acc1':(
            str(np.mean(np.array(valid_acc1_li))),
            str(np.std(np.array(valid_acc1_li)))),
        'valid_acc5':(
            str(np.mean(np.array(valid_acc5_li))),
            str(np.std(np.array(valid_acc5_li)))),
        'test_acc1':(
            str(np.mean(np.array(test_acc1_li))),
            str(np.std(np.array(test_acc1_li)))),
        'test_acc5':(
            str(np.mean(np.array(test_acc5_li))),
            str(np.std(np.array(test_acc5_li)))),
    }
    return acc_dict

def get_results_single_logdir(config, dataset_name, ext=''):
    pn_logs = os.path.join(get_logdir(config, dataset_name, ext=ext), 'logs.csv')
    df = pd.read_csv(pn_logs)
    if 'OpenML' in dataset_name:
        best_epoch = df[df['valid_acc_bal'] == max(df['valid_acc_bal'])]
        valid_acc1 = best_epoch.iloc[-1]['valid_acc_bal']
        valid_acc5 = best_epoch.iloc[-1]['valid_acc_bal']
        test_acc1 = best_epoch.iloc[-1]['test_acc_bal']
        test_acc5 = best_epoch.iloc[-1]['test_acc_bal']
    else:
        df_subset = df[df['valid_acc5'] == max(df['valid_acc5'])]
        best_epoch = df_subset[df_subset['valid_acc1'] == max(df_subset['valid_acc1'])]
        valid_acc1 = best_epoch.iloc[-1]['valid_acc1']
        valid_acc5 = best_epoch.iloc[-1]['valid_acc5']
        test_acc1 = best_epoch.iloc[-1]['test_acc1']
        test_acc5 = best_epoch.iloc[-1]['test_acc5']
    return valid_acc1, valid_acc5, test_acc1, test_acc5

def run_consistency(config, dataset_name, compression_li, ext=''):
    reset_weights = True
    if dataset_name in ['SVHN', 'CIFAR10', 'CIFAR100']:
        ckpt = 'checkpoint-40.pt'
        contrastive = '2,simclr_batchconstant'
    elif 'OpenML' in dataset_name:
        ckpt = 'checkpoint-105.pt'
        contrastive = '2,LoCL_batchconstant'
    else:
        assert False
    for compression in compression_li:
        exp = \
            ConsistencyExperiment(
                dataset=dataset_name,
                model=config['model'],
                strategy='GlobalMagWeight',
                resume=os.path.join(get_logdir(config, dataset_name, ext=ext), 'checkpoints', ckpt),
                path=get_logdir(config, dataset_name, ext=f'{ext}{DELIMITER}{compression}{DELIMITER}con'),
                compression=compression,
                reset_weights=reset_weights,
                pretrained=False,
                small_init_multiplier=config['small_init_multiplier'],
                train_kwargs={'epochs': 1},
                save_freq=99999,
                contrastive=contrastive,
                snapshot_ensemble=False,
                beta_lasso=None
            )

def run_visualization(config, dataset_name, ext=''):
    reset_weights = True
    if dataset_name in ['SVHN', 'CIFAR10', 'CIFAR100']:
        ckpt = 'checkpoint-40.pt'
    elif 'OpenML' in dataset_name:
        ckpt = 'checkpoint-105.pt'
    else:
        assert False
    exp = \
        VisualizationExperiment(
            dataset=dataset_name,
            model=config['model'],
            strategy='GlobalMagWeight',
            resume=os.path.join(get_logdir(config, dataset_name, ext=ext), 'checkpoints', ckpt),
            path=get_logdir(config, dataset_name, ext=f'{ext}{DELIMITER}vis'),
            compression=1.0,
            reset_weights=reset_weights,
            pretrained=False,
            small_init_multiplier=config['small_init_multiplier'],
            train_kwargs={'epochs': config['epochs']},
            save_freq=config['epochs'] - 1 if config['save_freq'] is None else config['save_freq'],
            contrastive=None,
            snapshot_ensemble=False,
            beta_lasso=None
        )

def run_train(config, dataset_name, ext=''):
    exp = \
        PruningExperiment(
            seed=42 if EXT == 'run0' else int(EXT[3:]),
            dataset=dataset_name,
            path=get_logdir(config, dataset_name, ext=ext),
            model=config['model'],
            strategy='GlobalMagWeight',
            resume=None,
            compression=1.0,
            reset_weights=False,
            pretrained=False,
            small_init_multiplier=config['small_init_multiplier'],
            train_kwargs={'epochs': config['epochs']},
            save_freq=config['epochs'] - 1 if config['save_freq'] is None else config['save_freq'],
            contrastive=config['contrastive'],
            snapshot_ensemble=config['snapshot_ensemble'], #True if 'OpenML' in dataset_name else None,
            beta_lasso=config['beta_lasso'] #  1e-6, 2e-6, 5e-6, 1e-5 and 2e-5
        )
    exp.run()

def run_finetune(config, dataset_name, ext=''):
    reset_weights = True#False#
    for compression in ([2,4] if 'OpenML' in dataset_name else [2,4,8,16]):#[1]:#
        if dataset_name in ['SVHN', 'CIFAR10', 'CIFAR100']:
            ckpt = 'checkpoint-40.pt'
        elif 'OpenML' in dataset_name:
            ckpt = 'checkpoint-105.pt'
        else:
            assert False
        print(f"loading from: {os.path.join(get_logdir(config, dataset_name, ext=ext), 'checkpoints', ckpt)}")
        exp = \
            PruningExperiment(
                seed=42 if EXT == 'run0' else int(EXT[3:]),
                dataset=dataset_name,
                model=config['model'],
                strategy='GlobalMagWeight',
                resume=os.path.join(get_logdir(config, dataset_name, ext=ext), 'checkpoints', ckpt),
                path=get_logdir(config, dataset_name, ext=f'{ext}{DELIMITER}{compression}{DELIMITER}ft'),
                compression=compression,
                reset_weights=reset_weights,
                pretrained=False,
                small_init_multiplier=None if config['small_init_multiplier'] is None else 1/config['small_init_multiplier'],
                train_kwargs={'epochs': config['epochs']},
                save_freq=config['epochs'] - 1,
                contrastive=None,
                snapshot_ensemble=config['snapshot_ensemble'],
                beta_lasso=None
            )
        exp.run()

if __name__ == '__main__':
    main()
