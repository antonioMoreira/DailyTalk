import typer

from dailytalk.cli_models import PreprocessArgs
from dailytalk.preprocessor import PreprocessorPipeline
from dailytalk.utils.tools import get_configs_of

app = typer.Typer(help="Unified Preprocessing Pipeline for DailyTalk.")


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of dataset"),
    stage: str = typer.Option(
        "all", "--stage", "-s", help="Stage to execute: 'all', 'preparation', 'language', 'features', 'speaker'"
    ),
    text: str | None = typer.Option(None, "--text", "-t", help="Sample text to process for 'language' stage"),
) -> None:
    if stage not in ("all", "preparation", "language", "features", "speaker"):
        raise typer.BadParameter("stage must be 'all', 'preparation', 'language', 'features', or 'speaker'")

    args = PreprocessArgs(dataset=dataset, stage=stage, text=text)  # type: ignore
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    pipeline = PreprocessorPipeline(preprocess_config, model_config, train_config)

    if args.stage == "preparation":
        res = pipeline.run_stage_1_preparation()
        typer.echo(f"Stage 1 (Preparation & Cleaning) complete: {res}")
    elif args.stage == "language":
        sample_text = args.text or "Hello world! This is DailyTalk text processing."
        seq = pipeline.run_stage_2_language_frontend(sample_text)
        typer.echo(f"Stage 2 (Language Frontend) complete for '{sample_text}':\nSymbol sequence IDs ({len(seq)}): {seq}")
    elif args.stage == "features":
        res_feat = pipeline.run_stage_3_feature_extraction()
        typer.echo(f"Stage 3 (Feature Extraction) complete: {res_feat}")
    elif args.stage == "speaker":
        count = pipeline.run_stage_4_speaker_embedding()
        typer.echo(f"Stage 4 (Speaker Embedding Extraction) complete: {count} embeddings")
    else:
        res_all = pipeline.run_all()
        typer.echo(f"Full Unified Preprocessing Pipeline complete: {res_all}")


if __name__ == "__main__":
    app()
