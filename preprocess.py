# Monkey-patch six to support PEP 451 under Python 3.14+
try:
    import six
    from importlib.machinery import ModuleSpec
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

from utils.tools import get_configs_of
from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    preprocessor = Preprocessor(preprocess_config, model_config, train_config)
    preprocessor.build_from_path()
