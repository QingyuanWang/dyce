
# DyCE: Dynamic Configurable Exiting for Deep Learning Compression and Scaling

Code release of [https://arxiv.org/abs/2403.01695]()

This repo contains a tool to search the optimal configuration of a multi-exit system, with different preferences between computation and accuracy.

For reproducing the expriment in the paper. Please see `examples`


## Usage

Step 1: Record predictions given by all exits.

```python
from dyce.utils import OutputRecorder

model.eval()
recorder = OutputRecorder(topk=1)
for images, targets in data_loader:
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True).unsqueeze(-1)
    # The output should be in the shape of (B, C) for the original head and (N, B, C) for all early exits.
    model_out, ee_out = model(images)
    recorder.add_pred(model_out, ee_out)
    recorder.add_label(targets)
recorder.save(save_path, meta_info=None)
```

Step 2: Search configurations for different $\lambda$ and evaluate the searched configuration on the validation set.

```python
from dyce.config_search import search_and_eval
def search_and_eval(
    sim_index='dyce_resnet50',
    macs_info=macs_info,
    train_record_path='path_to_train_record',
    val_record_path='path_to_val_record'
)
```

`macs_info` contains the complexity infomation of the backbone and all exits. `examples/macs_analysis.py` can generate required infomation for supported models. For customized models, please prepare in following format.

```python
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
```

Where `macs_info["head"][i]["macs"][j]` is the macs of the i-th type exit at j-th position, `macs_info["backbone"]["macs"][j]` is the accumlative macs of the backbone before entering the exit at j-th position. `macs_info["backbone"]["ori_macs"]` is the total macs of the original model.
