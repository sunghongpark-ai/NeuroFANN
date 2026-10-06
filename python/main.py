import sys

sys.dont_write_bytecode = True

from model.run_model import _cli, run_model


def main(data_source=None, overrides=None):
    return run_model(data_source, overrides)


if __name__ == "__main__":
    raise SystemExit(_cli())
