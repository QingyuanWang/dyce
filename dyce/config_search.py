import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from scipy.optimize import fminbound
import json
import scipy
import time
import os
import math
from .utils import print_info
class ConfigEval():
    '''
    Evaluate a the complexity and accuracy of a configurations.
    Implemented in numpy with straightforward logic. Please normally use ConfigEvalAccelerated.
    '''
    def __init__(self, label:np.ndarray, exit_preds:list[np.ndarray], exit_cons:list[np.ndarray], exit_macs:list[list[int]], bb_macs:list[int], base_acc:float, base_macs:float) -> None:
        '''
        Parameters:
            label - np.ndarray, the ground truth label in the shape of (N,)
            exit_preds - list[np.ndarray], the predictions of the exits at each position
            exit_cons - list[np.ndarray], the confidence of the exits at each position
            exit_macs - list[list[int]], the macs of the exits
            bb_macs - list[int], the macs of the backbone
            base_acc - float, the accuracy of the original model
            base_macs - float, the macs of the original model
        '''
        self.label = label
        self.exit_preds = exit_preds
        self.exit_cons = exit_cons
        self.exit_macs = exit_macs
        self.bb_macs = bb_macs
        self.num_of_sample = self.label.shape[0]
        self.num_of_exit = len(self.exit_preds)
        self.base_acc = base_acc
        self.base_macs = base_macs

        self.macs_min = bb_macs[0]
        self.macs_max = base_macs
        self.acc_min = (exit_preds[0][0][:, 0] == label.squeeze()).sum() / self.num_of_sample
        self.acc_max = base_acc
        print(
            f'Base acc: {self.base_acc}, base macs: {self.base_macs} max macs: {self.macs_max} min macs: {self.macs_min} max acc: {self.acc_max} min acc: {self.acc_min}'
        )

    def eval(self, config):
        exited = np.full((self.num_of_sample, 1), False)
        macs_enabled_exit_accum = 0
        macs_accum = 0
        correct_accum = 0
        for pos in range(len(config)):
            exit_sel = config[pos][0]
            if exit_sel == -1:
                continue
            else:
                sample_arrive = np.logical_not(exited)
                sample_exit = np.logical_and(
                    sample_arrive, self.exit_cons[pos][exit_sel][:, 0].reshape(-1, 1) > config[pos][1]
                )
                exited = np.logical_or(exited, sample_exit)
                correct_accum += self._correct_sample(sample_exit, self.exit_preds[pos][exit_sel]).sum()

                macs_enabled_exit_accum += self.exit_macs[pos][exit_sel]
                macs_accum += sample_exit.sum() / self.num_of_sample * (macs_enabled_exit_accum + self.bb_macs[pos])
                if exited.sum() == exited.shape[0]:
                    break
        correct_accum += self._correct_sample(np.logical_not(exited), self.exit_preds[-1][0]).sum()
        macs_accum += np.logical_not(exited).sum() / self.num_of_sample * (self.bb_macs[-1] + macs_enabled_exit_accum)
        return macs_accum / self.base_macs, correct_accum / self.num_of_sample / self.base_acc  # MAC, ACC

    def _correct_sample(self, exit_by_this, pred):
        correct = np.logical_and(exit_by_this, self.label == pred[:, 0].reshape(-1, 1))
        return correct


class ConfigEvalAccelerated(ConfigEval):
    '''
    Accelerated version of ConfigEval, implemented in pytorch.
    CEA allows update the existing config with only computing changed parts.
    '''

    def __init__(self, label:np.ndarray, exit_preds:list[np.ndarray], exit_cons:list[np.ndarray], exit_macs:list[list[int]], bb_macs:list[int], base_acc:float, base_macs:float, device:str|torch.device='cuda') -> None:
        '''
        See ConfigEval for details.
        '''
        super().__init__(label, exit_preds, exit_cons, exit_macs, bb_macs, base_acc, base_macs)
        torch.set_grad_enabled(False)
        print(f'Using device: {device}')
        self.device = device
        self.sample_index_vector = torch.arange(self.num_of_sample, device=self.device)

        self._to_torch(self.device)
        self.reset()

    def reset(self):
        '''
        Reset the enclosed configuration to the empty state. i.e. all samples go to the final exit.
        '''
        self.config = [(-1, 1) for _ in range(len(self.exit_cons) - 1)]
        self.exit_at_pos = torch.ones((self.num_of_sample,), device=self.device) * (self.num_of_exit - 1)
        self.correct_exit = (self.label == self.exit_preds[-1][0][:, 0].reshape(-1,))
        self.exit_macs_accum = torch.tensor(self.bb_macs, device=self.device).squeeze()
        self.pos_last = None
        self.exit_typei_last = None
        self.exit_at_pos_record = torch.ones(
            (self.num_of_sample, self.num_of_exit), device=self.device, dtype=torch.uint8
        ) * (
            self.num_of_exit - 1
        )
        self.exit_at_pos_record[:, -1] = self.num_of_exit - 1
        self.correct_exit_record = torch.zeros(
            (self.num_of_sample, self.num_of_exit), device=self.device, dtype=torch.bool
        )
        self.correct_exit_record[:, -1] = self.correct_exit

    def update_config(self, pos:int, exit_typei:int, t:float, save_change:bool=False):
        '''
        Update the configuration with one modification.
        Parameters:
            pos - int, the position of the exit.
            exit_typei - int, the type of the exit. -1 for no exit.
            t - float, [0,1], the confidence threshold of the exit.
            save_change - bool, whether to save the change. If false, return the result without saving the modification.
        '''
        correct_exit_temp = self.correct_exit.clone()
        exit_at_pos_temp = self.exit_at_pos.clone().byte()
        no_update = False
        if self.pos_last is None or pos != self.pos_last:
            update_pos = True
            self.pos_last = pos
        else:
            update_pos = False
        if update_pos or self.exit_typei_last is None or exit_typei != self.exit_typei_last:
            update_exit = True
            self.exit_typei_last = exit_typei
        else:
            update_exit = False

        if update_pos:
            self.sample_arrived_mask = self.exit_at_pos >= pos
            self.label_arrived = self.label[self.sample_arrived_mask]
            self.correct_exit_arrived = self.correct_exit[self.sample_arrived_mask]
            self.exit_at_pos_arrived = self.exit_at_pos[self.sample_arrived_mask]
            self.sample_arrived_count = self.sample_arrived_mask.sum()

        if self.sample_arrived_count != 0:

            if exit_typei != -1:
                if update_pos or update_exit:
                    self.con_arrived = torch.masked_select(
                        self.exit_cons[pos][exit_typei][:, 0], self.sample_arrived_mask
                    )
                    self.pred_arrived = torch.masked_select(
                        self.exit_preds[pos][exit_typei][:, 0], self.sample_arrived_mask
                    )
                exit_by_this_mask_arrived = self.con_arrived > t  #self.thres_ref[pos][exit_typei][int(t * 1000)][self.pred_arrived.int()]
                if update_pos or update_exit or (self.exit_by_this_mask_arrived_last == exit_by_this_mask_arrived
                                                ).sum() != exit_by_this_mask_arrived.shape[0]:
                    correct_exit_arrived_temp = self.correct_exit_arrived.clone()
                    exit_at_pos_arrived_temp = self.exit_at_pos_arrived.clone()
                    if exit_by_this_mask_arrived.sum() != 0:
                        # print(self.label_arrived.shape,self.sample_arrived_pred.shape,exit_by_this_mask_arrived.shape, exit_by_this_mask_arrived.sum())
                        exit_by_this_pred = torch.masked_select(self.pred_arrived, exit_by_this_mask_arrived)
                        exit_by_this_label = torch.masked_select(self.label_arrived, exit_by_this_mask_arrived)
                        correct_exit_arrived_temp[exit_by_this_mask_arrived] = (exit_by_this_pred == exit_by_this_label)
                        exit_at_pos_arrived_temp[exit_by_this_mask_arrived] = pos
                    # write back
                    correct_exit_temp[self.sample_arrived_mask] = correct_exit_arrived_temp
                    exit_at_pos_temp[self.sample_arrived_mask] = exit_at_pos_arrived_temp.byte()
                else:
                    no_update = True
                self.exit_by_this_mask_arrived_last = exit_by_this_mask_arrived

                # Use future exits if samples are not exited
                not_exit_mask = torch.logical_not(exit_by_this_mask_arrived)
            else:
                not_exit_mask = torch.ones((self.sample_arrived_mask.sum(),), device=self.device, dtype=torch.bool)

            if not no_update and not_exit_mask.sum() > 0:
                arrived_not_exit_mask = self.sample_arrived_mask.clone()
                arrived_not_exit_mask[self.sample_arrived_mask] = not_exit_mask
                future_exit_pos = torch.amin(self.exit_at_pos_record[arrived_not_exit_mask, pos + 1:], dim=-1)
                future_exit_pos_ = torch.zeros_like(arrived_not_exit_mask).long()
                future_exit_pos_[arrived_not_exit_mask] = future_exit_pos.long().clone()
                # exit_at_pos_temp = torch.where(arrived_not_exit_mask, future_exit_pos_, exit_at_pos_temp)
                exit_at_pos_temp[arrived_not_exit_mask] = future_exit_pos
                # correct_exit_temp[arrived_not_exit_mask] = self.correct_exit_record[arrived_not_exit_mask, future_exit_pos]
                correct_exit_temp = torch.where(
                    arrived_not_exit_mask, self.correct_exit_record[self.sample_index_vector,
                                                                    future_exit_pos_.long()], correct_exit_temp
                )
        exit_macs_accum_temp = self.exit_macs_accum.clone()
        if not no_update and self.config[pos][0] != -1:
            exit_macs_accum_temp[pos:] -= self.exit_macs[pos][self.config[pos][0]]
        if not no_update and exit_typei != -1:
            exit_macs_accum_temp[pos:] += self.exit_macs[pos][exit_typei]

        if no_update:
            exit_macs_accum_temp = self.exit_macs_accum_temp_last
            correct_exit_temp = self.correct_exit_temp_last
            exit_at_pos_temp = self.exit_at_pos_temp_last
        else:
            self.exit_macs_accum_temp_last = exit_macs_accum_temp
            self.correct_exit_temp_last = correct_exit_temp
            self.exit_at_pos_temp_last = exit_at_pos_temp
        if save_change:
            self.exit_at_pos_record[:, pos] = self.num_of_exit - 1
            if exit_typei == -1:
                self.correct_exit_record[:, pos] = False
            else:
                exit_by_this_mask_all = self.exit_cons[pos][exit_typei][:, 0] > t
                self.correct_exit_record[:, pos] = (self.label == self.exit_preds[pos][exit_typei][:, 0].reshape(-1,))
                self.exit_at_pos_record[exit_by_this_mask_all, pos] = pos

            self.correct_exit = correct_exit_temp
            self.exit_at_pos = exit_at_pos_temp
            self.exit_macs_accum = exit_macs_accum_temp
            self.config[pos] = [exit_typei, t]

        self.pos_last = pos
        self.exit_typei_last = exit_typei
        return correct_exit_temp, exit_at_pos_temp, exit_macs_accum_temp

    def eval(self, config=None, calibration=True, return_exit_at_pos=False):

        if config is None:
            return self._eval_current(calibration=calibration, return_exit_at_pos=return_exit_at_pos)
        else:
            for pos, (exit_typei, t) in enumerate(config):
                self.update_config(pos, exit_typei, t, save_change=True)
            return self._eval_current(calibration=calibration, return_exit_at_pos=return_exit_at_pos)

    def config_search_samethres(self, lam:float, **kwargs):
        '''
        Yield a configuration with the same threshold for all exits.
        lam - float, threshold for all exits.
        kwargs['exit_typei'] - int, the type index of the exit. Default: 0
        '''
        exit_typei = kwargs.pop('exit_typei', 0)
        return [(exit_typei, lam) for _ in range(len(self.exit_cons) - 1)]

    def config_search_circular(self, lam:float, max_rounds:int=1000, verbose:bool=False, calibration:bool=False, normalize:bool=False):
        '''
        Circular search for the optimal configuration with a given lambda.
        Parameters:
            lam - float, the lambda in the cost function
            max_rounds - int, the maximum rounds of circular search
            verbose - bool, whether to print the intermediate results
            calibration - bool, whether to calibrate the result with a consavative estimation. Only useful when the dataset is small.
            normalize - bool, whether to normalize the accuracy and complexity during the search. Default: false
        '''

        self.reset()
        rounds = 0
        cost_min = 1000
        # with tqdm(total=max_rounds) as pbar:
        while True:
            candidate = None
            for pos in range(len(self.exit_cons) - 1):
                for exit_typei in list(range(len(self.exit_cons[pos]))) + [-1]:
                    if exit_typei == -1:
                        t, cost = 1, self.cost_func(1, pos, exit_typei, lam, calibration=calibration, normalize=normalize)
                    else:
                        t, cost, _, nfev = fminbound(
                            self.cost_func, 0, 1, (pos, exit_typei, lam, False, False, calibration), full_output=True
                        )
                    if cost < cost_min:
                        cost_min = cost
                        candidate = (pos, exit_typei, t, cost)
            if candidate is not None:
                pos, exit_typei, t, cost = candidate
                if verbose:
                    print(rounds, pos, exit_typei, t, cost)
                self.update_config(pos, exit_typei, t, save_change=True)

            else:
                break
            rounds += 1
            if rounds == max_rounds:
                break
        return self.config.copy()

    def config_search_onepass(self, lam:float, verbose:bool=False, calibration:bool=False, normalize:bool=False):
        '''
        Onepass search for the optimal configuration with a given lambda.
        Parameters:
            lam - float, the lambda in the cost function
            verbose - bool, whether to print the intermediate results
            calibration - bool, whether to calibrate the result with a consavative estimation. Only useful when the dataset is small.
            normalize - bool, whether to normalize the accuracy and complexity during the search. Default: false
        '''
        self.reset()
        search_list = list(range(len(self.exit_cons) - 1))
        for pos in search_list:
            cost_min = 1000
            type_list = list(range(len(self.exit_cons[pos]))) + [-1]
            for exit_typei in type_list:
                if exit_typei == -1:
                    t, cost = 1, self.cost_func(1, pos, exit_typei, lam, calibration=calibration, normalize=normalize)
                else:
                    t, cost, _, nfev = fminbound(
                        self.cost_func,
                        0,
                        1, (pos, exit_typei, lam, False, normalize, calibration),
                        full_output=True,
                    )

                if cost <= cost_min:
                    cost_min = cost
                    candidate = (pos, exit_typei, t, cost)
                if verbose:
                    cost, acc, macs = self.cost_func(
                        t, pos, exit_typei, lam, aux_output=True, calibration=calibration, normalize=normalize
                    )
                    print(
                        f'pos:{pos} exit_typei:{exit_typei} t:{t} cost:{cost} acc:{acc} macs:{macs} cost_min:{cost_min} can:{candidate}'
                    )
            pos, exit_typei, t, cost = candidate
            if verbose:
                cost, acc, macs = self.cost_func(
                    t, pos, exit_typei, lam, aux_output=True, normalize=normalize, calibration=calibration
                )
                print(f'Update pos:{pos} exit_typei:{exit_typei} t:{t} cost{cost} acc:{acc} macs:{macs}')
                print('-' * 50)
            self.update_config(pos, exit_typei, t, save_change=True)
        return self.config.copy()

    def config_search(self, method, **kwargs):
        '''
        Search the optimal configuration. Return the configuration, the complexity and the accuracy.
        See each method for details
        '''
        if method == 'onepass':
            config = self.config_search_onepass(**kwargs)
        elif method == 'circular':
            config = self.config_search_circular(**kwargs)
        elif method == 'samethres':
            config = self.config_search_samethres(**kwargs)
        else:
            raise NotImplementedError
        macs, acc = self._eval_current()
        return config, macs, acc

    def _eval_current(
        self, correct_exit=None, exit_at_pos=None, exit_macs_accum=None, calibration=False, return_exit_at_pos=False
    ):
        '''
        Return the estimated complexity and accuracy of the current config.
        '''
        if correct_exit is None:
            correct_exit = self.correct_exit
        if exit_macs_accum is None:
            exit_macs_accum = self.exit_macs_accum
        if exit_at_pos is None:
            exit_at_pos = self.exit_at_pos
        macs_bar = (exit_macs_accum[exit_at_pos.long()].double() / self.num_of_sample).sum()
        exit_at_pos_ratio = []
        if not calibration:
            acc_bar = correct_exit.sum() / self.num_of_sample

            for i in range(exit_at_pos.max().long().item() + 1):
                exited = exit_at_pos.long() == i
                exited_count = exited.sum()
                exit_at_pos_ratio.append(exited_count.item() / self.num_of_sample)
        else:
            acc_bar = 0
            for i in range(exit_at_pos.max().long().item() + 1):
                exited = exit_at_pos.long() == i
                exited_count = exited.sum()
                correct_count = correct_exit[exited].sum()
                acc_calibrated = scipy.special.betaincinv(
                    1 + correct_count.item(), 1 + exited_count.item() - correct_count.item(), 0.1
                )
                acc_bar += acc_calibrated * exited_count / self.num_of_sample
                exit_at_pos_ratio.append(exited_count.item() / self.num_of_sample)

        if return_exit_at_pos:
            return (macs_bar / self.base_macs).item(), (acc_bar / self.base_acc).item(), exit_at_pos_ratio
        return (macs_bar / self.base_macs).item(), (acc_bar / self.base_acc).item()

    def update_and_eval(self, pos:int, exit_typei:int, t:float, save_change:bool=False, calibration:bool=False):
        '''
        Update the current config with one step, return the accuracy and complexity.
        Parameter:
            pos - the position where the change applies
            exit_typei - the new exit type index
            t - the new threshold
            save_change - whether to save the change. If false, return the result without saving the modification.
            calibration - whether to calibrate the result with a consavative estimation. Only useful when the dataset is small.
        '''
        correct_exit, exit_at_pos, exit_macs_accum = self.update_config(pos, exit_typei, t, save_change=save_change)
        macs_bar, acc_bar = self._eval_current(correct_exit, exit_at_pos, exit_macs_accum, calibration=calibration)
        return macs_bar, acc_bar

    def cost_func(self, t:float, pos:int, exit_typei:int, lam:float, aux_output=False, normalize=False, calibration=False):
        '''
        The cost fuction to minimize. The cost is a weighted by lambda between the complexity and the accuracy. 
        The threshold t is placed at the first parameter as the required by scipy.optimize.fminbound.
        Parameters:
            t - float, the threshold
            pos - int, the position of the exit
            exit_typei - int, the type of the exit
            lam - float, the weight of the accuracy
            aux_output - bool, whether to return the auxilary output, used for searching
            normalize - bool, whether to normalize the accuracy and complexity during the search. Default: false
            calibration - bool, whether to calibrate the result with a consavative estimation. Only useful when the dataset is small.

        '''
        correct_exit, exit_at_pos, exit_macs_accum = self.update_config(pos, exit_typei, t, save_change=False)
        macs_bar, acc_bar = self._eval_current(correct_exit, exit_at_pos, exit_macs_accum, calibration=calibration)
        if normalize:
            macs_bar = (macs_bar * self.base_macs - self.macs_min) / (self.macs_max - self.macs_min)
            acc_bar = (acc_bar * self.base_acc - self.acc_min) / (self.acc_max - self.acc_min)
        if aux_output:
            return (lam * (1 - acc_bar) + (1 - lam) * macs_bar), acc_bar, macs_bar
        else:
            return (lam * (1 - acc_bar) + (1 - lam) * macs_bar)

    def _to_torch(self, device: str | torch.device = 'cuda'):
        '''
        Convert the records to torch.Tensor and move to the device.
        '''
        memory_size = 0
        if isinstance(self.label, np.ndarray):
            self.label = torch.from_numpy(self.label).squeeze().to(device)
            memory_size += self.label.element_size() * self.label.nelement()
        exit_cons = []
        exit_preds = []
        for pos, pos_cons in enumerate(self.exit_cons):
            if isinstance(self.exit_preds[pos], np.ndarray):
                exit_preds.append(torch.from_numpy(self.exit_preds[pos]).to(device))
            elif isinstance(self.exit_preds[pos], torch.Tensor):
                exit_preds.append(self.exit_preds[pos].to(device))
            memory_size += exit_preds[-1].element_size() * exit_preds[-1].nelement()
            if isinstance(self.exit_cons[pos], np.ndarray):
                exit_cons.append(torch.from_numpy(self.exit_cons[pos]).to(device))
            elif isinstance(self.exit_cons[pos], torch.Tensor):
                exit_cons.append(self.exit_cons[pos].to(device))
            memory_size += exit_cons[-1].element_size() * exit_cons[-1].nelement()
        print(f'Used VRAM:{memory_size/(1024**3):.3f}Gb')

        self.exit_preds = exit_preds
        self.exit_cons = exit_cons



def load_records(record_path: str, macs_info: dict):
    '''
    Load the record generated by OutputRecorder
    Parameters:
        record_path - The path to the record
        macs_info - The complexity information of the model
            Example:
            {
                "head":
                    [
                        {
                            "name": "mlp1x1000",
                            "macs": [40000, 80000, 160000, 160000, 160000, 320000, 0],
                            "params": []
                        }, {
                            "name": "mlp3x500",
                            "macs": [770000, 790000, 830000, 830000, 830000, 910000, 0],
                            "params": []
                        }
                    ],
                "backbone":
                    {
                        "name": "convnextv2_atto",
                        "macs": [52308480, 151845120, 288449280, 371804160, 455159040, 506103360, 547332480],
                        "ori_macs": 547332480,
                    }
            }
            Where macs_info["head"][i]["macs"][j] is the macs of the i-th type exit at j-th position, 
            macs_info["backbone"]["macs"][j] is the accumlative macs of the backbone before entering the exit at j-th position.
            macs_info["backbone"]["ori_macs"] is the total macs of the original model.

    '''

    record_dict = np.load(record_path, allow_pickle=True)
    label = record_dict['label']
    preds = record_dict['preds'].astype(np.int16)
    confs = record_dict['confs']
    print(f'Load record from {record_path}, shape: {confs.shape}')
    if 'meta_info' in record_dict.keys():
        meta_info = record_dict['meta_info']
    else:
        meta_info = None
    bb_macs = macs_info['backbone']['macs']
    if 'shared' in macs_info:
        assert len(macs_info['backbone']['macs']) == len(macs_info['shared']['macs'])
        for i in range(len(macs_info['backbone']['macs'])):
            bb_macs[i]+= macs_info['shared']['macs'][i]
    exit_macs = []
    len_type = len(macs_info['head'])
    len_pos = math.ceil(confs.shape[0] / len_type)
    num_samples = label.shape[0]

    sd_info = []
    base_acc = (preds[-1][:, 0] == label.squeeze()).sum() / num_samples
    base_macs = macs_info['backbone']['ori_macs']
    preds_reorganized = []
    confs_reorganized = []
    for i in range(len_pos - 1):
        preds_reorganized.append(preds[i * len_type:(i + 1) * len_type])
        confs_reorganized.append(confs[i * len_type:(i + 1) * len_type])

    preds_reorganized.append(preds[[-1], :])
    confs_reorganized.append(confs[[-1], :])
    for i in range(len_pos):
        exit_macs.append([])
        sd_info.append([])
        for j in range(len(preds_reorganized[i])):
            exit_macs[i].append(macs_info['head'][j]['macs'][i])
            sd_macs = bb_macs[i] + exit_macs[i][j]
            sd_acc = (preds_reorganized[i][j][:, 0] == label.squeeze()).sum() / num_samples
            sd_info[i].append([sd_acc, sd_acc / base_acc * 100, sd_macs, sd_macs / base_macs * 100])
    return {
        'label': label,
        'preds': preds_reorganized,
        'confs': confs_reorganized,
        'meta_info': meta_info,
        'bb_macs': bb_macs,
        'exit_macs': exit_macs,
        'base_acc': base_acc,
        'base_macs': base_macs,
        'len_pos': len_pos,
        'len_type': len_type,
        'num_samples': num_samples,
        'sd_info': sd_info
    }




def save_search_result(
    cea:ConfigEvalAccelerated, configs: list, base_macs: int, base_acc: float, save_dir: str, save_name: str
):
    '''
    Get the simulated results of the searched configurations on the dataset enclosed with cea and save the results.
    Parameters:
        cea - ConfigEvalAccelerated, the object to evaluate the configurations
        configs - list, the configurations to be evaluated. Each entry contains a config generated by cea.config_search() and a lambda.
                e.g. {'exit_config':config, 'lambda':lam}
        base_macs - int, the macs of the original model
        base_acc - float, the accuracy of the original model
        save_dir - str, the directory to save the result
        save_name - str, the name of the result file
    '''
    searched_configs = []
    for config in tqdm(configs):
        cea.reset()
        macs_train, acc_train = config['exit_config'][1], config['exit_config'][2]
        macs, acc, exit_at_pos = cea.eval(config['exit_config'][0], calibration=False, return_exit_at_pos=True)
        searched_configs.append(
            [
                macs, (macs * base_macs), acc, (acc * base_acc), macs_train, macs_train * base_macs, acc_train,
                acc_train * base_acc, config, exit_at_pos, config['lambda']
            ]
        )
        # searched_configs.append([macs, acc, config, self.lams[i]])
    searched_configs.sort(key=lambda x: x[0])

    columns = [
        'relative_macs', 'raw_macs', 'relative_acc', 'raw_acc', 'relative_macs_tr', 'raw_macs_tr', 'relative_acc_tr',
        'raw_acc_tr', 'config', 'exit_ratio', 'lambda'
    ]

    figure_df = pd.DataFrame(searched_configs, columns=columns)
    searched_configs_filtered = [searched_configs[0]]
    for i in range(len(searched_configs) - 1):
        if searched_configs[i + 1][2] > searched_configs_filtered[-1][2]:
            searched_configs_filtered.append(searched_configs[i + 1])
    config_df_filtered = pd.DataFrame(searched_configs_filtered, columns=columns)

    if not os.path.exists(f'{save_dir}/unfiltered'):
        os.makedirs(f'{save_dir}/unfiltered', exist_ok=True)
    figure_df.to_csv(f'{save_dir}/unfiltered/{save_name}.csv')
    config_df_filtered.to_csv(f'{save_dir}/{save_name}.csv')
    print(f"[{save_name.split('_')[0]}] Finished save  to {save_dir}/{save_name}.csv")


def search_and_eval(
    sim_index: str,
    macs_info: str|dict,
    train_record_path: str,
    val_record_path: str,
    search_range: list[float] = [0.5, 1.01],
    search_step: float = 0.01,
    save_dir: str = './eval/sim_results/',
    save_name: str = None,
    method: str = 'onepass',
    calibration: bool = False,
    device: str = 'cuda',
    refresh:bool=False
):
    '''
    Search the optimal configuration with different lambda and save the simulated results.
    Parameters:
        sim_index - str, the index of the simulation
        macs_info - str|dict, the path to the macs_info or the dict of macs_info. See load_records for details.
        train_record_path - str, the path to the train record
        val_record_path - str, the path to the val record
        search_range - list[float], the range of candidate lambda
        search_step - float, the step of candidate lambda
        save_dir - str, the directory to save the result
        save_name - str, the name of the result file. Default: {sim_index}_{method}
        method - str, the search method. Options: 'onepass', 'circular', 'samethres'
        calibration - bool, whether to calibrate the result with a consavative estimation. Only useful when the dataset is small.
        device - str, the device to use. Default: 'cuda'
        refresh - bool, whether to refresh the result if the file exists. Default: False
    '''
    if save_name is None:
        save_name = f'{sim_index}_{method}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not refresh and os.path.exists(f'{save_dir}/{save_name}.csv'):
        print(f'{save_dir}/{save_name}.csv exists, skip')
        return
    if isinstance(macs_info, str):
        with open(macs_info, 'r') as f:
            macs_info = json.load(f)
    print(f'[{sim_index}] Start')
    print(f'[{sim_index}] Load train record from: {train_record_path}')
    record_info = load_records(train_record_path, macs_info)
    print_info(record_info['bb_macs'], record_info['base_macs'], record_info['sd_info'], sim_index=sim_index, split='train')
    cea = ConfigEvalAccelerated(
        record_info['label'], record_info['preds'], record_info['confs'], record_info['exit_macs'], record_info['bb_macs'], record_info['base_acc'], record_info['base_macs'], device=device
    )
    start_time = time.time()
    configs = []
    for lam in tqdm(np.arange(*search_range, search_step)):
        configs.append({'exit_config': cea.config_search(method, lam=lam, calibration=calibration, normalize=True),'lambda': lam})
    end_time = time.time()
    print(f'[{sim_index}] Conf generated')
    print(f'[{sim_index}] Load test record from: {val_record_path}')

    record_info = load_records(val_record_path, macs_info)
    print_info(record_info['base_acc'], record_info['base_macs'], record_info['sd_info'], sim_index=sim_index, split='test')

    cea = ConfigEvalAccelerated(
        record_info['label'], record_info['preds'], record_info['confs'], record_info['exit_macs'], record_info['bb_macs'], record_info['base_acc'], record_info['base_macs'], device=device
    )
    save_search_result(cea, configs, record_info['base_macs'], record_info['base_acc'], save_dir, save_name)
    print(f'[{sim_index}] Time cost: {end_time - start_time}')
