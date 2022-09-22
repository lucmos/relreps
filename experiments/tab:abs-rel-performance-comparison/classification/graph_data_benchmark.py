import functools
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from omegaconf import DictConfig
from sklearn.model_selection import ParameterGrid
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCN2Conv, GCNConv, GINConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import RandomNodeSplit
from tqdm import tqdm

from nn_core.common.utils import seed_index_everything

from rae import PROJECT_ROOT
from rae import lightning_logger as seed_log
from rae.modules.attention import AttentionOutput, RelativeAttention
from rae.modules.enumerations import Output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(dataset_name: str, num_anchors: int):
    transform = T.Compose([T.NormalizeFeatures(), RandomNodeSplit(num_val=0.1, num_test=0)])
    dataset = Planetoid(PROJECT_ROOT / "data" / "pyg" / dataset_name, dataset_name, transform=transform)
    data = dataset[0]
    _, edge_weight = gcn_norm(
        data.edge_index, num_nodes=data.x.size(0), add_self_loops=False
    )  # Pre-process GCN normalization.
    data.edge_weight = edge_weight
    data.anchors = torch.as_tensor(random.sample(data.train_mask.nonzero().squeeze().cpu().tolist(), num_anchors))

    return dataset, data


def encoder_factory(encoder_type, num_layers: int, in_channels: int, out_channels: int, **params):
    assert num_layers > 0
    if encoder_type == "GCN2Conv":
        convs = []
        for layer in range(num_layers):
            convs.append(GCN2Conv(layer=layer + 1, channels=out_channels, **params))
        return nn.ModuleList(convs)

    elif encoder_type == "GCNConv":
        convs = []
        # current_out_channels = in_channels
        #
        # for layer in range(num_layers):
        #     convs.append(
        #         GCNConv(
        #             in_channels=current_out_channels,
        #             out_channels=(current_out_channels := max(out_channels, current_out_channels // 2)),
        #             **params,
        #         )
        #     )
        convs = [
            GCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                **params,
            )
        ]
        in_channels = out_channels
        for layer in range(num_layers - 1):
            convs.append(
                GCNConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **params,
                )
            )
        return nn.ModuleList(convs)

    elif encoder_type == "GATConv":
        convs = []

        # for layer in range(num_layers):
        #     convs.append(
        #         GATConv(
        #             in_channels=current_out_channels,
        #             out_channels=(current_out_channels := max(out_channels, current_out_channels // 2)),
        #             **params,
        #         )
        #     )

        convs = [
            GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                **params,
            )
        ]
        in_channels = out_channels
        for layer in range(num_layers - 1):
            convs.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **params,
                )
            )

        return nn.ModuleList(convs)

    elif encoder_type == "GINConv":
        convs = []
        # current_out_channels = in_channels
        #
        # for layer in range(num_layers):
        #     convs.append(
        #         GINConv(
        #             nn=nn.Linear(
        #                 in_features=current_out_channels,
        #                 out_features=(current_out_channels := max(out_channels, current_out_channels // 2)),
        #             )
        #         )
        #     )
        current_in_channels = in_channels
        for layer in range(num_layers):
            convs.append(
                GINConv(
                    nn=nn.Linear(
                        in_features=current_in_channels,
                        out_features=out_channels,
                    ),
                    **params,
                )
            )
            current_in_channels = out_channels
        return nn.ModuleList(convs)

    else:
        raise NotImplementedError


class Net(torch.nn.Module):
    def __init__(
        self,
        hidden_proj: nn.Module,
        hidden_fn,
        relative_proj: RelativeAttention,
        class_proj: nn.Module,
        convs: nn.ModuleList,
        conv_fn,
        conv_out: int,
        dropout: float,
        relative: bool,
    ):
        super().__init__()

        self.hidden_proj: nn.Module = hidden_proj
        self.class_proj: nn.Module = class_proj

        self.hidden_fn = hidden_fn

        self.relative_proj = relative_proj

        self.convs = convs
        self.conv_fn = conv_fn
        self.conv_fc = nn.Linear(in_features=conv_out, out_features=conv_out)

        self.dropout = dropout
        self.relative: bool = relative

    def forward(self, x, edge_index, edge_weight, anchor_idxs: torch.Tensor):
        # TODO: batchnorm
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hidden_proj(x)

        x = x_0 = self.hidden_fn(x)

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            params = {"edge_index": edge_index}
            if type(self.convs[0]).__name__ == "GCN2Conv":
                params["x_0"] = x_0
                params["edge_weight"] = edge_weight
            x = conv(x, **params)
            x = self.conv_fn(x)

        x = self.conv_fc(x)
        anchors: torch.Tensor = x[anchor_idxs, :]

        rel_out = self.relative_proj(x=x, anchors=anchors)
        if self.relative:
            x = rel_out[AttentionOutput.OUTPUT]

        x = F.normalize(x, p=2, dim=-1)
        x = self.class_proj(x)

        return {Output.LOGITS: x, Output.SIMILARITIES: rel_out[AttentionOutput.SIMILARITIES]}


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, anchor_idxs=data.anchors)
    logits = out[Output.LOGITS]
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def model_test(model, data):
    model.eval()
    out = model(data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, anchor_idxs=data.anchors)
    pred = out[Output.LOGITS].argmax(dim=-1)

    accs = []
    for _, mask in data("train_mask", "val_mask"):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return out, accs


if __name__ == "__main__":
    dst_path: Path = PROJECT_ROOT / "experiments/tab:abs-rel-performance-comparison/classification/graph_benchmark.tsv"
    # df = pd.read_csv((PROJECT_ROOT / "test.tsv"), sep="\t")
    # print(df.groupby(["relative", "dataset"]).agg([np.mean, np.std]))
    # exit()

    # sweep = {
    #     # "seed_index": list(range(5)),
    #     "seed_index": [0],
    #     "num_epochs": [100],
    #     "in_channels": [64, 128, 256, 512],
    #     # "out_channels": [10, 32, 64],
    #     "out_channels": [num_anchors],
    #     "num_layers": [16, 32, 64],
    #     "dropout": [0.1, 0.7, 0.8],
    #     # "hidden_fn": [torch.relu, torch.tanh, torch.sigmoid],
    #     # "conv_fn": [torch.relu, torch.tanh, torch.sigmoid],
    #     "hidden_fn": [torch.nn.ReLU(), torch.nn.Tanh()],
    #     "conv_fn": [torch.nn.ReLU(), torch.nn.Tanh()],
    #     "optimizer": [torch.optim.Adam],
    #     "lr": [0.01, 0.001, 0.0001],
    #     "encoder": [
    #         (
    #             "GCN2Conv",
    #             functools.partial(
    #                 encoder_factory,
    #                 encoder_type="GCN2Conv",
    #                 **dict(alpha=0.1, theta=0.5, shared_weights=True, normalize=False),
    #             ),
    #         ),
    #         # ("GCNConv", functools.partial(encoder_factory, encoder_type="GCNConv")),
    #         # ("GATConv", functools.partial(encoder_factory, encoder_type="GATConv")),
    #         # ("GINConv", functools.partial(encoder_factory, encoder_type="GINConv")),
    #     ],
    #     "relative": [True],
    # }
    num_anchors: int = 300

    sweep = {
        "seed_index": list(range(10)),
        "dataset": ["Cora", "PubMed", "CiteSeer"],
        "num_epochs": [500],
        "in_channels": [num_anchors],
        "out_channels": [num_anchors],
        "num_layers": [32, 64],
        "num_anchors": [num_anchors],
        "dropout": [0.5],
        "hidden_fn": [torch.nn.ReLU()],
        "conv_fn": [torch.nn.ReLU()],
        "optimizer": [torch.optim.Adam],
        "lr": [0.02],
        "encoder": [
            (
                "GCN2Conv",
                functools.partial(
                    encoder_factory,
                    encoder_type="GCN2Conv",
                    **dict(alpha=0.1, theta=0.5, shared_weights=True, normalize=False),
                ),
            ),
            # ("GCNConv", functools.partial(encoder_factory, encoder_type="GCNConv")),
            # ("GATConv", functools.partial(encoder_factory, encoder_type="GATConv")),
            #         ("GINConv", functools.partial(encoder_factory, encoder_type="GINConv")),
        ],
        "relative": [True, False],
    }

    # keys, values = zip(*sweep.items())
    # experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    experiments = ParameterGrid(sweep)
    print(f"Total available experiments={len(experiments)}")

    # for i, experiment in enumerate(pbar := tqdm(experiments, desc="Experiment")):
    if not dst_path.exists():
        with dst_path.open("w", encoding="utf-8") as fw:
            keys = list(sweep.keys())
            keys.append("val_acc")
            header: str = "\t".join(keys)
            fw.write(f"{header}\n")

            for experiment in (pbar := tqdm(experiments, desc="Experiment")):
                # pprint(experiment)
                temp_log_level = seed_log.getEffectiveLevel()
                seed_log.setLevel(logging.ERROR)
                seed_index_everything(DictConfig({"seed_index": experiment["seed_index"]}))
                seed_log.setLevel(temp_log_level)

                dataset_name = experiment["dataset"]
                num_anchors: int = experiment["num_anchors"]

                dataset, data = get_data(dataset_name=dataset_name, num_anchors=num_anchors)

                encoder_name, encoder_build = experiment["encoder"]
                if encoder_name == "GCN2Conv":
                    experiment["out_channels"] = num_anchors
                    experiment["in_channels"] = num_anchors

                hidden_proj = nn.Linear(dataset.num_features, experiment["in_channels"])

                convs = encoder_build(
                    num_layers=experiment["num_layers"],
                    in_channels=experiment["in_channels"],
                    out_channels=experiment["out_channels"],
                )
                class_proj = nn.Linear(experiment["out_channels"], dataset.num_classes)

                model = Net(
                    hidden_proj=hidden_proj,
                    hidden_fn=experiment["hidden_fn"],
                    relative_proj=RelativeAttention(
                        n_anchors=num_anchors,
                        n_classes=dataset.num_classes,
                        similarity_mode="inner",
                        values_mode="similarities",
                        normalization_mode="l2",
                    ),
                    class_proj=class_proj,
                    convs=convs,
                    conv_fn=experiment["conv_fn"],
                    conv_out=experiment["out_channels"],
                    dropout=experiment["dropout"],
                    relative=experiment["relative"],
                ).to(DEVICE)
                data = data.to(DEVICE)
                optimizer = experiment["optimizer"](
                    model.parameters(),
                    lr=experiment["lr"],
                )

                best_val_acc = 0
                best_epoch = None
                epochs = []
                for epoch in range(experiment["num_epochs"]):
                    loss = train_step(model=model, optimizer=optimizer, data=data)
                    model_out, (train_acc, val_acc) = model_test(model=model, data=data)
                    # epochs.append(epoch_out)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_epoch = {
                            "rel_x": model_out[Output.SIMILARITIES].to("cpu", non_blocking=True),
                            "epoch": epoch,
                            "loss": loss,
                            "train_acc": train_acc,
                            "val_acc": val_acc,
                        }
                    # print(
                    #     f"Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, "
                    #     f"Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, "
                    #     f"Final Test: {test_acc:.4f}"
                    # )
                experiment["best_epoch"] = best_epoch
                # experiment["epochs"] = epochs
                # best_epoch = epochs[best_epoch]
                pbar.set_description(
                    f"Epoch: {best_epoch['epoch']:04d}, Loss: {best_epoch['loss']:.4f} Train: {best_epoch['train_acc']:.4f}, "
                    f"Val: {best_epoch['val_acc']:.4f}"
                )

                items = []
                for key in sweep.keys():
                    run_value = experiment[key]
                    if key == "encoder":
                        run_value = run_value[0]
                    elif "_fn" in key:
                        run_value = type(run_value).__name__
                    elif key == "optimizer":
                        run_value = run_value.__name__
                    items.append(run_value)
                items.append(best_epoch["val_acc"])

                line: str = "\t".join(map(str, items))
                fw.write(f"{line}\n")
                fw.flush()

                model.cpu()

    df = pd.read_csv(dst_path, sep="\t")
    df = (
        df.drop(columns=[col for col in df.columns if col not in {"val_acc", "dataset", "relative"}])[df.lr == 0.02]
        .groupby(["relative", "dataset"])
        .agg([np.mean, np.std])
    )
    print(df)
