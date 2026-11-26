import typer

from dailytalk.cli_models import PrepareAlignArgs
from dailytalk.preprocessor import dailytalk
from dailytalk.utils.tools import get_configs_of

app = typer.Typer(help="Prepare alignment dataset for DailyTalk.")


def prepare_align(config):
    if "DailyTalk" in config["dataset"]:
        dailytalk.prepare_align(config)


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of dataset"),
):
    args = PrepareAlignArgs(dataset=dataset)
    config, *_ = get_configs_of(args.dataset)
    prepare_align(config)


if __name__ == "__main__":
    app()
