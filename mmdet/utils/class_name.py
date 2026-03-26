from mmengine.utils import is_str

dataset_aliases = {
    'wuhanUV': {'wuhanUV'},
    'cityscapes': ['cityscapes'],
    'ade': ['ade', 'ade20k'],
    'voc': ['voc', 'pascal_voc', 'voc12', 'voc12aug'],
    'pcontext': ['pcontext', 'pascal_context', 'voc2010'],
    'loveda': ['loveda'],
    'potsdam': ['potsdam'],
    'vaihingen': ['vaihingen'],
    'cocostuff': [
        'cocostuff', 'cocostuff10k', 'cocostuff164k', 'coco-stuff',
        'coco-stuff10k', 'coco-stuff164k', 'coco_stuff', 'coco_stuff10k',
        'coco_stuff164k'
    ],
    'isaid': ['isaid', 'iSAID'],
    'stare': ['stare', 'STARE'],
    'lip': ['LIP', 'lip'],
    'mapillary_v1': ['mapillary_v1'],
    'mapillary_v2': ['mapillary_v2'],
    'bdd100k': ['bdd100k'],
    'hsidrive': [
        'hsidrive', 'HSIDrive', 'HSI-Drive', 'hsidrive20', 'HSIDrive20',
        'HSI-Drive20'
    ]
}

def get_classes(dataset):
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels


def get_palette_seg(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_palette()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels