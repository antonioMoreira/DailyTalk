import typer

from dailytalk.cli_models import PreprocessArgs
from dailytalk.preprocessor import PreprocessorPipeline
from dailytalk.utils.tools import get_configs_of

app = typer.Typer(help="Unified Preprocessing Pipeline for DailyTalk.")


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of dataset"),
    stage: str = typer.Option(
        "all", "--stage", "-s", help="Stage to execute: 'all', 'preparation', 'features', 'speaker'"
    ),
):
    args = PreprocessArgs(dataset=dataset)
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    pipeline = PreprocessorPipeline(preprocess_config, model_config, train_config)

    if stage == "preparation":
        res = pipeline.run_stage_1_preparation()
        typer.echo(f"Stage 1 (Preparation & Cleaning) complete: {res}")
    elif stage == "features":
        res_feat = pipeline.run_stage_3_feature_extraction()
        typer.echo(f"Stage 3 (Feature Extraction) complete: {res_feat}")
    elif stage == "speaker":
        count = pipeline.run_stage_4_speaker_embedding()
        typer.echo(f"Stage 4 (Speaker Embedding Extraction) complete: {count} embeddings")
    else:
        res_all = pipeline.run_all()
        typer.echo(f"Full Unified Preprocessing Pipeline complete: {res_all}")


if __name__ == "__main__":
    app()
