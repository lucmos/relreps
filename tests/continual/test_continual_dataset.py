import random
from collections import Counter, defaultdict

import pytest
from pytorch_lightning import Trainer

from nn_core.common import PROJECT_ROOT

from rae.data.continual.cifar10 import ContinualCIFAR10Dataset


@pytest.mark.parametrize(
    "dataset_class, dataset_kwargs",
    (
        (
            ContinualCIFAR10Dataset,
            {
                "path": PROJECT_ROOT / "data",
                "transform": None,
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "tasks_epochs, tasks_progression",
    (
        (
            [5, 5, 5],
            [[1, 3, 6], [2, 4, 6], [4, 8, 9]],
        ),
        (
            [2, 9, 4],
            [[1, 3, 6], [2, 4, 6], [4, 8, 9]],
        ),
        (
            [1, 1, 2, 1],
            [[1, 3, 6], [2], [4, 8, 9], [0]],
        ),
        (
            [1, 1, 2, 1],
            [[0], [2], [1], [0]],
        ),
    ),
)
def test_continual_dataset(dataset_class, dataset_kwargs, tasks_epochs, tasks_progression):
    trainer: Trainer = lambda x: x
    trainer.current_epoch = 0

    dataset = dataset_class(
        split="train",
        tasks_epochs=tasks_epochs,
        tasks_progression=tasks_progression,
        trainer=trainer,
        **dataset_kwargs,
    )

    for _ in range(5):
        task2retrieved_targets = defaultdict(list)
        task2epochs_done = Counter()
        for epoch in range(15):
            trainer.current_epoch = epoch
            for k in range(20):
                sample = dataset[random.randint(0, len(dataset) - 1)]
                task2retrieved_targets[dataset.current_task].append(sample["target"])

            task2epochs_done[dataset.current_task] += 1

        # Test correct number of epochs per task
        for task_idx in range(len(tasks_progression) - 1):
            assert task2epochs_done[task_idx] == tasks_epochs[task_idx]

        # Test correct targets per tasks
        for task_idx in range(len(tasks_progression)):
            assert set(task2retrieved_targets[task_idx]) == set(tasks_progression[task_idx])
