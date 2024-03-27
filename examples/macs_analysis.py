import torch
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
from models.dmc import MultiExitNetowrk
import json
from models.utils import build_model
from functools import partial
import os
from tqdm import tqdm
import logging


@torch.no_grad()
def get_complexity(
    model_name,
    input_size=(3, 224, 224),
    save_path=None,
    refresh=False,
    device='cuda',
):
    logging.disable(logging.INFO)
    dmc_class = MultiExitNetowrk
    dmc_params = {
        'num_classes': 1000,
        'input_size': input_size,
    }
    if save_path is None:
        os.makedirs('./eval/macs_info', exist_ok=True)
        save_path = f'./eval/macs_info/dyce_{model_name}.json'
    if not refresh and os.path.exists(save_path):
        print(f'Macs info of {model_name} exists: {save_path}')
        with open(save_path, 'r') as f:
            macs_dict = json.load(f)
        return macs_dict
    print('Macs info not exists, generating...')
    inp = torch.rand((1, *input_size)).to(device)
    create_model = partial(build_model, model_name, device, strict_dict_load=False)
    model = create_model(interm_feat=True).to(device)
    model.eval()
    prof = FlopsProfiler(model)
    prof.start_profile()

    _, interm_feat = model(inp)
    if 'allowed_exits' in dmc_params and dmc_params['allowed_exits'] is not None:
        for i in range(len(dmc_params['allowed_exits'])):
            if dmc_params['allowed_exits'][i] < 0:
                dmc_params['allowed_exits'][i] += len(interm_feat)

    flops_ori = prof.get_total_flops(as_string=False)
    macs_ori = prof.get_total_macs(as_string=False)
    params_ori = prof.get_total_params(as_string=False)
    # prof.print_model_profile()
    prof.end_profile()
    print(f'GMacs: {macs_ori/(10**9):.2f}, GParams: {params_ori/(10**9):.2f}')

    macs_dict = {}
    macs_dict['head'] = []
    macs_dict['backbone'] = {
        'name': model_name,
        'macs': [],
        'macs_ratio': [],
        'params': [],
        'params_ratio': [],
        'ori_macs': macs_ori,
        'ori_params': params_ori
    }
    model_main = create_model(interm_feat=True, profiling=10000).to(device)
    model = dmc_class(model_main, **dmc_params).to(device)
    model.eval()
    exit_type_config = model.get_exit_type_config()
    exit_positions = list(range(len(interm_feat) - 1))

    for i in tqdm(exit_positions, desc='Profiling'):
        # print(len(interm_feat), exit_positions)
        model.model.profiling = i + 2

        prof = FlopsProfiler(model)
        prof.start_profile()
        out, interm_feats = model.forward_profiling_backbone(inp)
        macs = prof.get_total_macs(as_string=False)

        # prof.print_model_profile()
        prof.end_profile()
        macs_dict['backbone']['macs'].append(macs)
        macs_dict['backbone']['macs_ratio'].append(macs / macs_ori * 100)
        for exit_typei, exit_type in tqdm(enumerate(exit_type_config), desc='Profiling'):
            if i == 0:
                macs_dict['head'].append({'name': exit_type['name'], 'macs': [], 'params': []})

            prof.start_profile()
            single_exit_out = model.forward_profiling_head(i, exit_typei, interm_feats)
            macs = prof.get_total_macs(as_string=False)
            prof.end_profile()
            macs_dict['head'][exit_typei]['macs'].append(macs)
            if i == exit_positions[-1]:
                macs_dict['head'][exit_typei]['macs'].append(0)
    macs_dict['backbone']['macs'].append(macs_ori)
    macs_dict['backbone']['macs_ratio'].append(macs_ori / macs_ori * 100)

    print(macs_dict['backbone'])
    print('len:', len(macs_dict['backbone']['macs']))

    print(save_path)
    with open(save_path, 'w') as f:
        json.dump(macs_dict, f, indent=4)
    return macs_dict
