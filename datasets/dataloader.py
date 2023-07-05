from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader


def _collate(data):
    tts_data_collated = defaultdict(list)
    tts_data_len_collated = defaultdict(list)

    for tts_data_i, tts_data_len_i in data:
        for k, v in tts_data_i.items():
            tts_data_collated[k].append(v)
        for k, v in tts_data_len_i.items():
            tts_data_len_collated[k].append(v)

    tts_data = dict()
    for k, v in tts_data_collated.items():
        tts_data[k] = nn.utils.rnn.pad_sequence(v, batch_first=True)

    tts_data_len = dict()
    for k, v in tts_data_len_collated.items():
        tts_data_len[k] = torch.stack(v).squeeze(1)

    return tts_data, tts_data_len


def ProsodyDataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=None,
    shuffle=None,
    prefetch_factor=2,
    persistent_workers=False,
    drop_last=True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate if batch_size > 1 else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
