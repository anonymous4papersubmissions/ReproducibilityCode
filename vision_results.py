import yaml
import os
import pandas as pd
import numpy as np
from pprint import pprint

from experiment import PruningExperiment
from experiment.visualizations import VisualizationExperiment
from experiment.consistency import ConsistencyExperiment

PLACEHOLDER = 'PLACEHOLDER'
DELIMITER = '_-_'

PATH_ROOT = None
os.environ['DATAPATH'] = None

EXT_LI = ['run0', 'run1', 'run2']
EXT = 'run0'

def get_config_li(config_method_pn_li, dataset_pn='CIFAR10.yaml', model_pn='MLP_CV.yaml'):
    config_pn_li = []
    for config_method_pn in config_method_pn_li:
        config_pn_li.append((
            os.path.join(PATH_ROOT, 'dataset', dataset_pn),
            os.path.join(PATH_ROOT, 'method', config_method_pn),
            os.path.join(PATH_ROOT, 'model', model_pn)))

    config_li = []
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
        config['save_freq'] = None
        config_li.append(config)
    return config_li

def get_config_method_pn_li(method):
    method_pn_li = []
    ext_li = []
    if method == 'mlp':
        method_pn_li.append('SUP.yaml')
        ext_li.append(f'{PLACEHOLDER}')
    elif method == 'omp':
        method_pn_li.append('SUP.yaml')
        for compression in [2,4,8,16]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'betalasso':
        method_pn_li.extend([
            'betalasso1e-6.yaml', 
            'betalasso2e-6.yaml', 
            'betalasso5e-6.yaml', 
            'betalasso1e-5.yaml', 
            'betalasso2e-5.yaml'
        ])
        ext_li.append(f'{PLACEHOLDER}')
    elif method == 'our_no_pis':
        method_pn_li.extend([
            'SUP_SCL_CV.yaml',
        ])
        for compression in [2,4,8,16]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'our_125_no_ilo':
        method_pn_li.extend([
            'SUP_pi_0_125.yaml', 
        ])
        for compression in [2,4,8,16]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'our_125_no_prune':
        method_pn_li.extend([
            'SUP_SCL_pi_0_125_CV.yaml', 
        ])
        ext_li.extend([f'{PLACEHOLDER}'])
    elif method == 'our_0625_no_ilo':
        method_pn_li.extend([
            'SUP_pi_0_0625.yaml', 
        ])
        for compression in [2,4,8,16]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'our_0625_no_prune':
        method_pn_li.extend([
            'SUP_SCL_pi_0_0625_CV.yaml', 
        ])
        ext_li.extend([f'{PLACEHOLDER}'])
    elif method == 'our':
        method_pn_li.extend([
            'SUP_SCL_pi_0_0625_CV.yaml',
            'SUP_SCL_pi_0_125_CV.yaml', 
            'SUP_SCL_pi_0_25_CV.yaml', 
            'SUP_SCL_pi_0_5_CV.yaml', 
        ])
        for compression in [2,4,8,16]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    else:
        assert False
    return method_pn_li, ext_li

def get_perf_wrapper(method, dataset_name='CIFAR10', model_name='MLP_CV'):
    method_pn_li, ext_li = get_config_method_pn_li(method)
    config_li = get_config_li(method_pn_li, dataset_pn=f'{dataset_name}.yaml', model_pn=f'{model_name}.yaml')
    perf_best = {'valid_acc1':[float('-inf'),0], 'test_acc1':[float('-inf'),0]}
    for ext in ext_li:
        for config in config_li:
            perf = get_results(config, dataset_name, ext_li=[ext.replace(f'{PLACEHOLDER}', run) for run in EXT_LI])
            if perf['valid_acc1'][0] != 'nan':
                perf_best = max(perf_best, perf, key=lambda x: float(x['valid_acc1'][0]))
    return float(perf_best['test_acc1'][0])*100, float(perf_best['test_acc1'][1])*100

def main():
    print('mlp')
    for dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        mlp_perf = get_perf_wrapper('mlp', dataset)
        omp_perf = get_perf_wrapper('omp', dataset)
        betalasso_perf = get_perf_wrapper('betalasso', dataset)
        our_perf = get_perf_wrapper('our', dataset)
        cnn_perf = get_perf_wrapper('mlp', dataset, 'CNN')
        perf_li = [mlp_perf,omp_perf,betalasso_perf,our_perf,cnn_perf]
        best = max([x[0] for x in perf_li])
        print(f'{dataset} &', ' & '.join([('{\\bf '+f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}'+'}' if x[0] == best else f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}') for x in perf_li]), '\\\\')

    print('resnet')
    for dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        mlp_perf = get_perf_wrapper('mlp', dataset, 'resnet')
        omp_perf = get_perf_wrapper('omp', dataset, 'resnet')
        betalasso_perf = get_perf_wrapper('betalasso', dataset, 'resnet')
        our_perf = get_perf_wrapper('our', dataset, 'resnet')
        perf_li = [mlp_perf,omp_perf,betalasso_perf,our_perf]
        best = max([x[0] for x in perf_li])
        print(f'{dataset} &', ' & '.join([('{\\bf '+f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}'+'}' if x[0] == best else f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}') for x in perf_li]), '\\\\')

    print('ablation')
    for dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
        if 'CIFAR10' == dataset:
            best_pis = '0625'
        else:
            best_pis = '125'
        mlp_perf = get_perf_wrapper('mlp', dataset)
        our_no_prune = get_perf_wrapper(f'our_{best_pis}_no_prune', dataset)
        our_no_pis = get_perf_wrapper(f'our_{best_pis}_no_ilo', dataset)
        our_no_ilo = get_perf_wrapper('our_no_pis', dataset)
        our_perf = get_perf_wrapper('our', dataset, 'resnet')
        perf_li = [mlp_perf,our_no_prune,our_no_pis,our_no_ilo,our_perf]
        best = max([x[0] for x in perf_li])
        print(f'{dataset} &', ' & '.join([('{\\bf '+f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}'+'}' if x[0] == best else f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}') for x in perf_li]), '\\\\')

def get_logdir(config, dataset_name, ext=''):
    #os.system(f"mkdir {os.path.join(PATH_ROOT, 'logs')}")
    dn = \
        f"{dataset_name}{DELIMITER}{config['model']}{DELIMITER}{config['small_init_multiplier']}{DELIMITER}" + \
        f"{config['contrastive']}{DELIMITER}{config['beta_lasso']}{DELIMITER}{ext}"
    pn = os.path.join(PATH_ROOT, 'logs', dn)
    return pn

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
            pass
            #print(f'missing directory: {dataset_name, config, ext}')
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
    idx = 0
    if 'OpenML' in dataset_name:
        best_epoch = df[df['valid_acc_bal'] == max(df['valid_acc_bal'])]
        valid_acc1 = best_epoch.iloc[idx]['valid_acc_bal']
        valid_acc5 = best_epoch.iloc[idx]['valid_acc_bal']
        test_acc1 = best_epoch.iloc[idx]['test_acc_bal']
        test_acc5 = best_epoch.iloc[idx]['test_acc_bal']
    else:
        df_subset = df[df['valid_acc5'] == max(df['valid_acc5'])]
        best_epoch = df_subset[df_subset['valid_acc1'] == max(df_subset['valid_acc1'])]
        valid_acc1 = best_epoch.iloc[idx]['valid_acc1']
        valid_acc5 = best_epoch.iloc[idx]['valid_acc5']
        test_acc1 = best_epoch.iloc[idx]['test_acc1']
        test_acc5 = best_epoch.iloc[idx]['test_acc5']
    return valid_acc1, valid_acc5, test_acc1, test_acc5

if __name__ == '__main__':
    main()
