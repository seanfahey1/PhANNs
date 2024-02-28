#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import toml


def _get_config_args():
    parser = argparse.ArgumentParser(description="PhANNs configuration handler.")
    parser.add_argument(
        "--c",
        "-config file",
        type=str,
        default=None,
        nargs="?",
        help="Config file in .toml format.",
    )
    parser.add_argument(
        "--s",
        "-set",
        type=_valid_s_k_v,
        default=None,
        nargs="*",
        help="Config file in .toml format.",
    )

    args = parser.parse_args()
    return args


def _valid_s_k_v(s: str) -> str:
    try:
        assert len(s.split(".")) == 3
        return s
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a valid format (section.key.value): {s!r}"
        )


def _get_config_path():
    return Path(__file__).parent.resolve() / "config/config.toml"


def load_stored_config():
    config_path = _get_config_path()
    if not config_path.is_file():
        config = clear_config()

    config = toml.load(config_path)
    if len(config) == 0:
        config = clear_config()

    return config


def _set_from_toml(path):
    new_config = toml.load(path)
    config = load_stored_config()

    for section, values in new_config.items():
        for key, value in values.items():
            if section not in config.keys():
                config[section] = dict()
            config[section][key] = value

    return config


def _set_from_cmd(section: str, key: str, value, config: dict) -> dict:
    if section not in config.keys():
        config["section"] = dict()

    config[section][key] = value

    return config


def write_config():
    config = load_stored_config()
    args = _get_config_args()
    if args.c is not None:
        config = _set_from_toml(Path(args.c))

    if args.s is not None:
        for item in args.s:
            s, k, v = item.split(".")
            config = _set_from_cmd(s, k, v, config)

    config_path = _get_config_path()

    with open(config_path, "w") as out:
        toml.dump(config, out)


def view_config():
    config = load_stored_config()
    for section, values in config.items():
        for key, value in values.items():
            print(f"{section: <16} -- {key: <24}: {value}")


def clear_config():
    config_path = _get_config_path()

    config = {
        "main": {"project_root_dir": "."},
        "file_locations": {
            "fasta_data_dir": "../../datasets/no_expert_curation_subset_OTH/grouped_fastas",
            "array_data_dir": "../../datasets/no_expert_curation_subset_OTH/loaded_data",
            "model_dir": "../../datasets/no_expert_curation_subset_OTH/models/",
            "test_outputs_dir": "../../datasets/no_expert_curation_subset_OTH/predictions/",
        },
        "class_info": {
            "group_prefixes": [
                "1_",
                "2_",
                "3_",
                "4_",
                "5_",
                "6_",
                "7_",
                "8_",
                "9_",
                "10_",
                "11_",
            ]
        },
        "class_numbers": {
            "BPL": 1,
            "CLR": 2,
            "HTJ": 3,
            "m_CP": 4,
            "MCP": 5,
            "m_TL": 6,
            "MTL": 7,
            "PTL": 8,
            "TFR": 9,
            "TSH": 10,
            "TSP": 11,
            "OTH": 12,
        },
        "group_names": {
            "BPL": "BPL",
            "CLR": "CLR",
            "HTJ": "HTJ",
            "m_CP": "m_CP",
            "MCP": "MCP",
            "m_TL": "m_TL",
            "MTL": "MTL",
            "PTL": "PTL",
            "TFR": "TFR",
            "TSH": "TSH",
            "TSP": "TSP",
            "OTH": "OTH",
        },
        "models": {
            "di_sc": False,
            "di_sc_p": False,
            "tri_sc": False,
            "tri_sc_p": False,
            "tetra_sc": False,
            "tetra_sc_p": False,
            "di": False,
            "di_p": False,
            "tri": False,
            "tri_p": False,
            "tetra_sc_tri_p": False,
            "all": True,
        },
        "test": {"model": "all"},
    }

    with open(config_path, "w") as out:
        toml.dump(config, out)

    return config


def verify_clear_config():
    if (
        input(
            "You are about to delete the existing config object. Enter 'y' to continue: "
        ).lower()
        != "y"
    ):
        clear_config()
