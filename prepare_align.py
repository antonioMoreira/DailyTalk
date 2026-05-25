# Monkey-patch six to support PEP 451 under Python 3.14+
try:
    from importlib.machinery import ModuleSpec

    import six
    # pyrefly: ignore [missing-attribute]
    if not hasattr(six._SixMetaPathImporter, "find_spec"):
        def find_spec(self, fullname, path, target=None):
            if fullname in self.known_modules:
                return ModuleSpec(fullname, self, is_package=self.is_package(fullname))
            return None
        # pyrefly: ignore [missing-attribute]
        six._SixMetaPathImporter.find_spec = find_spec
except Exception:
    pass

import argparse

from preprocessor import dailytalk
from utils.tools import get_configs_of


def main(config):
    if "DailyTalk" in config["dataset"]:
        dailytalk.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config, *_ = get_configs_of(args.dataset)
    main(config)
