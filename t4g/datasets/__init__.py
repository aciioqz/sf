from t4g.data import BenchDataset


from t4g.datasets.bio import BioDataset, FullBioDataset, BaseBioDataset

dataset_cls_list = [
    BioDataset, FullBioDataset, BaseBioDataset
]

dataset_cls_dict = {dataset_cls.name: dataset_cls for dataset_cls in dataset_cls_list}

dataset_names = list(dataset_cls_dict.keys())


def get_dataset(name: str, *args, **kwargs) -> BenchDataset:
    r"""Returns a dataset by name."""
    print(f'get_dataset')
    return dataset_cls_dict[name](*args, **kwargs)

