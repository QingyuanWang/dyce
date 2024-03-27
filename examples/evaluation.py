import os
try:
    from dyce.config_search import search_and_eval
except ImportError:
    import sys
    sys.path.append('..')
    from dyce.config_search import search_and_eval
from macs_analysis import get_complexity
import multiprocessing as mp
import torch
import traceback
import sys
import argparse
from utils import model_abbr_name
import datetime
from utils import str2bool


def evaluation(sim_index, args, refresh=False, device='cuda'):
    train_record = f'{args.record_path}/{sim_index}/{sim_index}_train.npz'
    val_record = f'{args.record_path}/{sim_index}/{sim_index}_test.npz'
    model_name = sim_index.replace('dyce_', '')
    macs_info = get_complexity(model_name, device=device)
    if not os.path.exists(f'{args.record_path}/{sim_index}'):
        print(f'{args.record_path}/{sim_index} not exists, abort.')
        return
    print(f'Evaluation for {args.record_path}/{sim_index}')

    search_and_eval(
        sim_index=sim_index,
        macs_info=macs_info,
        train_record_path=train_record,
        val_record_path=val_record,
        search_step=0.001,
        search_range=[0.5, 1.05],
        refresh=refresh,
        device=device,
        save_dir=f'{args.eval_path}/sim_results',
    )


def evaluation_mp(arg):
    sim_index, args = arg
    rank = mp.current_process()._identity[0] - 1
    time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    fp = open(f'{args.eval_path}/sim_results/log/{time}-{sim_index}_{rank}.log', "a")
    sys.stdout = fp
    sys.stderr = fp
    device = 'cpu' if args.cpu else f'cuda:{rank}'
    print(sim_index, device)
    try:
        evaluation(sim_index, args, device=device)
    except:
        print('Error for', sim_index)
        traceback.print_exc()
    finally:
        fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_path', type=str, default='./records')
    parser.add_argument('--eval_path', type=str, default='./eval')
    parser.add_argument('--cpu', type=str2bool, default=False)
    parser.add_argument('--nproc', type=int, default=-1)
    args = parser.parse_args()
    sims_ = os.listdir(args.record_path)

    os.makedirs(f'{args.eval_path}/sim_results/log', exist_ok=True)
    sims = []
    for sim in sims_:
        if not os.path.exists(f'{args.record_path}/{sim}/{sim}_train.npz'):
            print(f'Record {args.record_path}/{sim}/{sim}_train.npz not exists, abort.')
        elif not os.path.exists(f'{args.record_path}/{sim}/{sim}_test.npz'):
            print(f'Record {args.record_path}/{sim}/{sim}_test.npz not exists, abort.')
        else:
            print(f'Evaluation for {args.record_path}/{sim}')
            sims.append((sim, args))
    mp.set_start_method('spawn')
    if args.nproc == -1:
        args.nproc = torch.cuda.device_count()
    with mp.Pool(args.nproc) as p:
        p.map(evaluation_mp, sims)
