# Marks configs/ as a package so setuptools ships the per-language YAML +
# _schema.json as package data. The files are loaded by path at runtime via
# core.config_loader (Path(__file__).parent.parent / "configs"), not imported.
