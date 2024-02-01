import argparse


def args():
    parser = argparse.ArgumentParser(
        description=" Generic inputs for PhANNs load, train, and test steps"
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="Relative path to a valid config .toml file",
    )
    parser.add_argument(
        "-v",
        "--value",
        type=str,
        required=False,
        nargs="*",
        help="Overwrite any configuration with a string formatted as section:key:value. These values are processed "
        "after the config file loads and are logged as if they were in the original config file.",
    )
    return parser.parse_args()
