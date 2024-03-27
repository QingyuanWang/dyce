# DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling

Reproducing the expriments of [https://arxiv.org/abs/2403.01695]()

Attaching simple exits to an existing system, training on ImageNet and configuraing with DyCE.


## Training

Step 1: Edit `accelerator_config.yaml` to match your local system

Step 2: Edit `task_config.yaml` for the path to ImageNet.

Step 3: Run training, e.g.

```bash
python run_task_parallel.py --tasks_in_parallel 4 --gpu_per_task 2
```

This will run the training of 4 base models at the same time and each task is assigned by 2 gpus. This requires your system to have 8 gpus in total.

The default `task_config.yaml` contains a task of exprimenting with `convnextv2_atto`, you can add more entries and `run_task_parallel.py` with execute them in sequence. e.g.

```yaml
tasks:
  - params:
      batch_size: 1024
      epoch: 50
      model: convnextv2_atto
      sim_index: dyce_convnextv2_atto
  - params:
      batch_size: 1024
      epoch: 50
      model: convnextv2_base
      sim_index: dyce_convnextv2_base
  - params:
      batch_size: 1024
      epoch: 50
      model: resnet50
      sim_index: dyce_resnet50
  - params:
      batch_size: 1024
      epoch: 50
      model: resnet152
      sim_index: dyce_resnet152
```

## Configuration searching & Evaluation

```bash
python evaluation.py
```

The results can be found in `eval/sim_results` by default.
