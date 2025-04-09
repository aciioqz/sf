import os

import pandas as pd
import pooch

from t4g.data import Database, BenchDataset, Table
from t4g.task.bio import RegressTask
from t4g.util.utils import to_unix_time, unzip_processor


def drop_column(table_df, users_drop_columns):
    for column in users_drop_columns:
        try:
            table_df.drop(columns=[column], inplace=True)
        except KeyError:
            pass

def remove_duplicates(s):
    items = s.split('|')
    unique_items = sorted(set(items), key=items.index)
    return '|'.join(unique_items)

class BioDataset(BenchDataset):
    name = "bio"
    task_cls_list = [RegressTask]

    def __init__(
            self,
            *,
            process: bool = False,
            path: str
    ):
        self.name = f"{self.name}"
        self.path = path
        super().__init__(process=process, path=path)

    def make_db(self) -> Database:
        r"""load data from local path."""
        # path = '/Users/caoziqi/Downloads/data/rel-benchmark/fake/raw'
        # 服务器路径
        path = self.path

        base = pd.read_csv(os.path.join(path, "base.csv"))
        base = base.drop(columns=['molecule_id'])
        atom = pd.read_csv(os.path.join(path, "atom.csv"))
        atom = atom.drop(columns=['atom_id','molecule_id'])
        bond = pd.read_csv(os.path.join(path, "bond.csv"))
        bond = bond.drop(columns=['atom_id','atom_id2'])
        gmember = pd.read_csv(os.path.join(path, "gmember.csv"))
        gmember = gmember.drop(columns=['atom_id','group_id'])

        group = pd.read_csv(os.path.join(path, "group.csv"))
        group = group.drop(columns=['group_id'])


        tables = {}

        tables["base"] = Table(
            df=pd.DataFrame(base),
            fkey_col_to_pkey_table={
            },
            pkey_col="id",
        )

        tables["atom"] = Table(
            df=pd.DataFrame(atom),
            fkey_col_to_pkey_table={
                "fpk_molecule_id": "base",
            },
            pkey_col="id",
            is_need_edge=False
        )

        tables["bond"] = Table(
            df=pd.DataFrame(bond),
            fkey_col_to_pkey_table={
                "fpk_atom_id": "atom",
                "fpk_atom_id2": "atom",
            },
            pkey_col="id",
            is_need_edge=False

        )

        tables["gmember"] = Table(
            df=pd.DataFrame(gmember),
            fkey_col_to_pkey_table={
                "fpk_group_id": "group",
                "fpk_atom_id": "atom",
            },
            pkey_col="id",
            is_need_edge=False

        )

        tables["group"] = Table(
            df=pd.DataFrame(group),
            fkey_col_to_pkey_table={
            },
            pkey_col="id",
            is_need_edge=False

        )
        return Database(tables)


class FullBioDataset(BenchDataset):
    name = "full-bio"
    task_cls_list = [RegressTask]

    def __init__(
            self,
            *,
            process: bool = False,
            path: str
    ):
        self.name = f"{self.name}"
        self.path = path
        super().__init__(process=process, path=path)

    def make_db(self) -> Database:
        r"""load data from local path."""
        # path = '/Users/caoziqi/Downloads/data/rel-benchmark/fake/raw'
        # 服务器路径
        path = self.path
        base = pd.read_csv(os.path.join(path, "base.csv"))
        base = base.drop(columns=['molecule_id'])
        atom = pd.read_csv(os.path.join(path, "atom.csv"))
        atom = atom.drop(columns=['atom_id', 'molecule_id'])
        bond = pd.read_csv(os.path.join(path, "bond.csv"))
        bond = bond.drop(columns=['atom_id', 'atom_id2'])
        gmember = pd.read_csv(os.path.join(path, "gmember.csv"))
        gmember = gmember.drop(columns=['atom_id', 'group_id'])

        group = pd.read_csv(os.path.join(path, "group.csv"))
        group = group.drop(columns=['group_id'])

        base = pd.merge(base, atom, left_on='id', right_on='fpk_molecule_id', how="left", suffixes=('', '_atom'))
        base = pd.merge(base, bond, left_on='id_atom', right_on='fpk_atom_id', how="left", suffixes=('', '_bond'))
        base = pd.merge(base, gmember, left_on='id_atom', right_on='fpk_atom_id', how="left", suffixes=('', '_gmember'))
        base = pd.merge(base, group, left_on='fpk_group_id', right_on='id', how="left", suffixes=('', '_group'))

        base = base.drop(columns=['id'])
        base.insert(0,'id',range(len(base)))


        tables = {}

        tables["base"] = Table(
            df=pd.DataFrame(base),
            fkey_col_to_pkey_table={
            },
            pkey_col="id",
        )
        return Database(tables)


class BaseBioDataset(BenchDataset):
    name = "base-bio"
    task_cls_list = [RegressTask]

    def __init__(
            self,
            *,
            process: bool = False,
            path: str
    ):
        self.name = f"{self.name}"
        self.path = path
        super().__init__(process=process, path=path)

    def make_db(self) -> Database:
        r"""load data from local path."""
        # path = '/Users/caoziqi/Downloads/data/rel-benchmark/fake/raw'
        # 服务器路径
        path = self.path

        base = pd.read_csv(os.path.join(path, "base.csv"), quotechar='"')

        tables = {}

        tables["base"] = Table(
            df=pd.DataFrame(base),
            fkey_col_to_pkey_table={
            },
            pkey_col="id",
        )
        return Database(tables)
