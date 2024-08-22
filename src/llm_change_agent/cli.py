"""Command line interface for llm-change-agent."""

import logging

import click

from llm_change_agent import __version__
from llm_change_agent.llm_agent import LLMChangeAgent
from llm_change_agent.utils.llm_utils import (
    get_anthropic_models,
    get_lbl_cborg_models,
    get_ollama_models,
    get_openai_models,
)

ALL_AVAILABLE_PROVIDERS = ["openai", "ollama", "anthropic", "cborg"]
ALL_AVAILABLE_MODELS = get_openai_models() + get_ollama_models() + get_anthropic_models() + get_lbl_cborg_models()

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """
    CLI for llm-change-agent.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)


@main.command()
def list_models():
    """List the available language models."""
    openai_models = "\n  ".join(get_openai_models())
    anthropic_models = "\n  ".join(get_anthropic_models())
    ollama_models = "\n  ".join(get_ollama_models())
    lbl_cborg_models = "\n  ".join(get_lbl_cborg_models())

    click.echo(f"OpenAI models:\n  {openai_models}")
    click.echo(f"Anthropic models:\n  {anthropic_models}")
    click.echo(f"Ollama models:\n  {ollama_models}")
    click.echo(f"LBL-CBORG models:\n  {lbl_cborg_models}")


@main.command()
@click.option("--model", type=click.Choice(ALL_AVAILABLE_MODELS), help="Model to use for generation.")
@click.option("--provider", type=click.Choice(ALL_AVAILABLE_PROVIDERS), help="Provider to use for generation.")
@click.option("--prompt", type=str, default="Hello, world!", help="Prompt to use for generation.")
def execute(model: str, provider: str, prompt: str):
    """Generate text using the specified model."""
    llm_agent = LLMChangeAgent(model=model, prompt=prompt, provider=provider)
    llm_agent.run()


if __name__ == "__main__":
    main()
