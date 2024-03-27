from collections import OrderedDict
import torch
import numpy as np
import os
import pandas as pd


class OutputRecorder():

    def __init__(self, topk=5):
        self.topk = topk
        self.initilize()

    def initilize(self):
        self.records = OrderedDict()
        self.records['label'] = []
        self.records['preds'] = []
        self.records['confs'] = []

    def add_pred(self, model_out: torch.Tensor, ee_out: torch.Tensor = None):
        '''
        model_out: The logits output of the original model in the shape of (B, C)
        ee_out: The logits output of the all early exits in the shape of (N, B, C)
        '''
        model_out = model_out.detach()
        if ee_out is not None:
            ee_out = ee_out.detach()
            out = torch.cat([ee_out, model_out.unsqueeze(1)], dim=1).permute([1, 0, 2])  # (N+1), B, C
        else:
            out = model_out.unsqueeze(1).permute([1, 0, 2])  # (N+1), B, C
        prob = out.softmax(dim=-1)
        confs, preds = torch.topk(prob, self.topk, dim=-1)  # (N+1), B, topk
        self.records['preds'].append(preds.detach().to('cpu').numpy().astype(np.int16))
        self.records['confs'].append(confs.detach().to('cpu').numpy())

    def add_label(self, label: torch.Tensor):
        '''
        label: The ground truth label in the shape of (B, )
        '''
        self.records['label'].append(label.view(-1, 1).detach().to('cpu').numpy().astype(np.int16))

    def save(self, path: str, meta_info: dict = None):
        '''
        path: The path to save the record
        meta_info: The meta information to save with the record
        '''
        print(f'Save record to {path}')
        path_dir = os.path.dirname(path)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        to_save = {}
        to_save['label'] = np.vstack(self.records['label'])
        to_save['confs'] = np.concatenate(self.records['confs'], axis=1)
        to_save['preds'] = np.concatenate(self.records['preds'], axis=1)
        if meta_info is not None:
            to_save['meta_info'] = np.array(meta_info)
        np.savez_compressed(path, **to_save)
        return to_save


def print_info(base_acc, base_macs, sd_info, sim_index=None, split='train'):
    if sim_index is not None:
        print(f'[{sim_index} {split}] ')
    print('\t Acc\t Acc%\t MACs\t\t MACs%')
    for i in range(len(sd_info)):
        print(f'Pos {i}')
        for j in range(len(sd_info[i])):
            print(f'\t{sd_info[i][j][0]:.4f}\t{sd_info[i][j][1]:.4f}\t{sd_info[i][j][2]:<10}\t{sd_info[i][j][3]:.4f}')
    print(f'Base acc: {base_acc}, base macs: {base_macs}')


def save_info(sd_info, save_path):
    info = []
    for i in range(len(sd_info)):
        for j in range(len(sd_info[i])):
            info.append(sd_info[i][j])
    info = pd.DataFrame(info, columns=['acc', 'acc%', 'macs', 'macs%'])
    info.to_csv(save_path, index=False)
