import typer

from dailytalk.cli_models import PreprocessArgs
from dailytalk.preprocessor.preprocessor import Preprocessor
from dailytalk.utils.tools import get_configs_of

app = typer.Typer(help="Preprocess dataset for DailyTalk.")


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of dataset"),
):
    args = PreprocessArgs(dataset=dataset)
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    preprocessor = Preprocessor(preprocess_config, model_config, train_config)
    preprocessor.build_from_path()


if __name__ == "__main__":
    app()
