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

openmlid2dataset_nm = \
    {
        'OpenML233088': 'credit-g',
        'OpenML233090': 'anneal',
        'OpenML233091': 'kr-vs-kp',
        'OpenML233092': 'arrhythmia',
        'OpenML233093': 'mfeat.',
        'OpenML233094': 'vehicle',
        'OpenML233096': 'kc1',
        'OpenML233099': 'adult',
        'OpenML233102': 'walking.',
        'OpenML233103': 'phoneme',
        'OpenML233104': 'skin-seg.',
        'OpenML233106': 'ldpa',
        'OpenML233107': 'nomao',
        'OpenML233108': 'cnae',
        'OpenML233109': 'blood.',
        'OpenML233110': 'bank.',
        'OpenML233112': 'connect.',
        'OpenML233113': 'shuttle',
        'OpenML233114': 'higgs',
        'OpenML233115': 'australian',
        'OpenML233116': 'car',
        'OpenML233117': 'segment',
        'OpenML233118': 'fashion.',
        'OpenML233119': 'jungle.',
        'OpenML233120': 'numerai',
        'OpenML233121': 'devnagari',
        'OpenML233122': 'helena',
        'OpenML233123': 'jannis',
        'OpenML233124': 'volkert',
        'OpenML233126': 'miniboone',
        'OpenML233130': 'apsfailure',
        'OpenML233131': 'christine',
        'OpenML233132': 'dilbert',
        'OpenML233133': 'fabert',
        'OpenML233134': 'jasmine',
        'OpenML233135': 'sylvine',
        'OpenML233137': 'dionis',
        'OpenML233142': 'aloi',
        'OpenML233143': 'ccfraud',
        'OpenML233146': 'clickpred.'
    }
openmlid2baseline = \
    {
        'OpenML233088': (68.929, 61.19, 74.643),
        'OpenML233090': (85.416, 84.248, 89.27),
        'OpenML233091': (99.85, 93.25, 99.85),
        'OpenML233092': (48.779, 43.562, 61.461),
        'OpenML233093': (98.0, 97.25, 98.0),
        'OpenML233094': (74.973, 79.654, 82.576),
        'OpenML233096': (66.846, 52.517, 74.381),
        'OpenML233099': (79.824, 77.155, 82.443),
        'OpenML233102': (61.616, 56.801, 63.923),
        'OpenML233103': (87.972, 86.824, 86.619),
        'OpenML233104': (99.968, 99.961, 99.953),
        'OpenML233106': (99.008, 54.815, 68.107),
        'OpenML233107': (96.872, 95.425, 96.826),
        'OpenML233108': (94.907, 89.352, 95.833),
        'OpenML233109': (62.281, 64.327, 67.617),
        'OpenML233110': (72.658, 70.639, 85.993),
        'OpenML233112': (72.374, 72.045, 80.073),
        'OpenML233113': (98.563, 88.017, 99.948),
        'OpenML233114': (72.944, 72.036, 73.546),
        'OpenML233115': (89.717, 85.278, 87.088),
        'OpenML233116': (92.376, 98.701, 99.587),
        'OpenML233117': (93.723, 91.775, 93.723),
        'OpenML233118': (91.243, 89.793, 91.95),
        'OpenML233119': (87.325, 73.425, 97.471),
        'OpenML233120': (52.363, 51.599, 52.668),
        'OpenML233121': (93.31, 94.179, 98.37),
        'OpenML233122': (21.994, 19.032, 27.701),
        'OpenML233123': (55.225, 56.214, 65.287),
        'OpenML233124': (64.17, 59.409, 71.667),
        'OpenML233126': (94.024, 62.173, 94.015),
        'OpenML233130': (88.825, 51.444, 92.535),
        'OpenML233131': (74.815, 69.649, 74.262),
        'OpenML233132': (99.106, 97.608, 99.049),
        'OpenML233133': (70.098, 62.277, 69.183),
        'OpenML233134': (80.546, 76.69, 79.217),
        'OpenML233135': (95.509, 83.595, 94.045),
        'OpenML233137': (91.222, 83.96, 94.01),
        'OpenML233142': (95.338, 93.589, 97.175),
        'OpenML233143': (90.303, 85.705, 92.531),
        'OpenML233146': (58.361, 50.163, 64.28)
    }


def get_config_li(config_method_pn_li):
    config_pn_li = []
    for config_method_pn in config_method_pn_li:
        config_pn_li.append((
            os.path.join(PATH_ROOT, 'dataset', 'OpenML.yaml'),
            os.path.join(PATH_ROOT, 'method', config_method_pn),
            os.path.join(PATH_ROOT, 'model', 'MLP_TAB.yaml')))

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
        for compression in [8]:
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
            'SUP_SCL_TAB.yaml',
        ])
        for compression in [8]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'our_no_ilo':
        method_pn_li.extend([
            'SUP_pi_0_25.yaml', 
        ])
        for compression in [8]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    elif method == 'our_no_prune':
        method_pn_li.extend([
            'SUP_SCL_pi_0_25_TAB.yaml', 
        ])
        ext_li.extend([f'{PLACEHOLDER}'])
    elif method == 'our':
        method_pn_li.extend([
            'SUP_SCL_pi_0_25_TAB.yaml', 
        ])
        for compression in [8]:
            ext_li.extend([f'{PLACEHOLDER}{DELIMITER}{compression}{DELIMITER}ft'])
    else:
        assert False
    return method_pn_li, ext_li

def get_perf_wrapper(method, dataset_name):
    method_pn_li, ext_li = get_config_method_pn_li(method)
    config_li = get_config_li(method_pn_li)
    best_config = None
    perf_best = {'valid_acc1':[float('-inf'),0], 'test_acc1':[float('-inf'),0]}
    for ext in ext_li:
        for config in config_li:
            perf = get_results(config, dataset_name, ext_li=[ext.replace(f'{PLACEHOLDER}', run) for run in EXT_LI])
            if perf['valid_acc1'][0] != 'nan':
                best_config = config 
                perf_best = max(perf_best, perf, key=lambda x: float(x['valid_acc1'][0]))
    return ((float(perf_best['test_acc1'][0])*100,float(perf_best['test_acc1'][1])*100), config)

def main():
    def get_str_ours(x, best):
        return '{\\bf '+f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}'+'}' if x[0] == best else f'{x[0]:.3f}'+' $\\pm$ '+f'{x[1]:.3f}'
    def get_str_base(x, best):
        return '{\\bf'+f'{x[0]:.3f}'+'}' if x[0] == best else f'{x[0]:.3f}'

    '''
    dataset2pis = {}
    for dataset, dataset_nm in sorted(openmlid2dataset_nm.items()):
        mlp_perf, _ = get_perf_wrapper('mlp', dataset)
        omp_perf, _ = get_perf_wrapper('omp', dataset)
        betalasso_perf, cfg_betalasso = get_perf_wrapper('betalasso', dataset)
        our_perf, cfg_our = get_perf_wrapper('our', dataset)
        #print(dataset, cfg_betalasso['beta_lasso'])
        dataset2pis[dataset] = int(100*cfg_our['small_init_multiplier'])
        baseline_perf = [(x,None) for x in openmlid2baseline[dataset]]
        perf_li = [mlp_perf, omp_perf, betalasso_perf, our_perf] + list(baseline_perf)
        best = max([x[0] for x in perf_li])
        #if perf_li[0] <= 90:
        #    continue
        print(f'{dataset_nm} &', '&'.join([(get_str_base(x, best) if (idx < 1 or idx >= len(perf_li)-len(baseline_perf)) else get_str_ours(x, best)) for idx, x in enumerate(perf_li)]), '\\\\')
    '''

    #for dataset in ['OpenML233092','OpenML233093','OpenML233094','OpenML233096']:
    for dataset, dataset_nm in sorted(openmlid2dataset_nm.items()):
        mlp_perf, _ = get_perf_wrapper('mlp', dataset)
        pru_perf, _ = get_perf_wrapper('our_no_prune', dataset)
        ilo_perf, _ = get_perf_wrapper('our_no_ilo', dataset)
        pis_perf, _ = get_perf_wrapper('our_no_pis', dataset)
        our_perf, _ = get_perf_wrapper('our', dataset)
        perf_li = [mlp_perf, pru_perf, ilo_perf, pis_perf, our_perf]
        best = max([x[0] for x in perf_li])
        print(f'{openmlid2dataset_nm[dataset]} &', '&'.join([get_str_ours(x, best) for x in perf_li]), '\\\\')

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
