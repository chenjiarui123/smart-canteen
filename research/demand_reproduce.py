"""Fresh full-day deployed-baseline reproduction; compare immutable old cache."""
import argparse
import shutil
import pandas as pd
import structural_prepare
from demand_data import OUT,PREVIOUS,verify


def main(dataset):
    verify()
    destination=OUT/"reproduction"
    destination.mkdir(exist_ok=True)
    config=destination/"effective_baseline_config.json"
    if not config.exists():shutil.copy2(PREVIOUS/"effective_baseline_config.json",config)
    structural_prepare.OUT=destination
    structural_prepare.prepare(dataset)
    name="development_"+dataset
    for artifact in ["baseline_menu.pkl","base_features.pkl"]:
        pd.testing.assert_frame_equal(pd.read_pickle(destination/name/artifact),pd.read_pickle(PREVIOUS/name/artifact),check_exact=True)
    verify()
    print(dataset,"fresh baseline exactly reproduced",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("dataset");main(p.parse_args().dataset)
