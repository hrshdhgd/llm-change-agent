"""Constants for the LLM Change Agent."""

from os import getenv

from importlib_metadata import files

OPENAI_KEY = str(getenv("OPENAI_API_KEY"))
ANTHROPIC_KEY = str(getenv("ANTHROPIC_API_KEY"))
CBORG_KEY = str(getenv("CBORG_API_KEY"))

OPEN_AI_MODEL = "gpt-4o-2024-08-06"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
OLLAMA_MODEL = "llama3.1"  #!  not all models support tools (tool calling)
CBORG_MODEL = "lbl/llama-3"

KGCL_SCHEMA = [file for file in files("kgcl-schema") if file.stem == "kgcl" and file.suffix == ".yaml"][0]
KGCL_GRAMMAR = [file for file in files("kgcl-schema") if file.stem == "kgcl" and file.suffix == ".lark"][0]

ONTODIFF_DOCS = [
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/EnvironmentOntology_envo/data_with_changes.yaml",
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/geneontology_go-ontology/data_with_changes.yaml",
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/monarch-initiative_mondo/data_with_changes.yaml",
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/obophenotype_cell-ontology/data_with_changes.yaml",
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/obophenotype_uberon/data_with_changes.yaml",
    "https://raw.githubusercontent.com/hrshdhgd/ontodiff-curator/main/pato-ontology_pato/data_with_changes.yaml",
]
