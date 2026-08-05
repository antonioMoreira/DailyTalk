# Monkey-patch six to support PEP 451 under Python 3.14+
try:
    from importlib.machinery import ModuleSpec

    import six

    if not hasattr(six._SixMetaPathImporter, "find_spec"):
        def find_spec(self, fullname, path, target=None):
            if fullname in self.known_modules:
                return ModuleSpec(fullname, self, is_package=self.is_package(fullname))
            return None

        six._SixMetaPathImporter.find_spec = find_spec  # type: ignore
except Exception:
    pass
