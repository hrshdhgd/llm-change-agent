"""Constants for the LLM Change Agent."""

from os import getenv

import pystow
from importlib_metadata import files

OPENAI_KEY = str(getenv("OPENAI_API_KEY"))
ANTHROPIC_KEY = str(getenv("ANTHROPIC_API_KEY"))
CBORG_KEY = str(getenv("CBORG_API_KEY"))

OPEN_AI_MODEL = "gpt-4o-2024-08-06"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
OLLAMA_MODEL = "llama3.1"  #!  not all models support tools (tool calling)
CBORG_MODEL = "anthropic/claude-sonnet"

OPENAI_PROVIDER = "openai"
ANTHROPIC_PROVIDER = "anthropic"
OLLAMA_PROVIDER = "ollama"
CBORG_PROVIDER = "cborg"

PROVIDER_DEFAULT_MODEL_MAP = {
    OPENAI_PROVIDER: OPEN_AI_MODEL,
    ANTHROPIC_PROVIDER: ANTHROPIC_MODEL,
    OLLAMA_PROVIDER: OLLAMA_MODEL,
    CBORG_PROVIDER: CBORG_MODEL,
}

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

ONTOLOGIES_AS_DOC_MAP = {
    "envo": "https://purl.obolibrary.org/obo/envo.json",
    "go": "https://purl.obolibrary.org/obo/go/go-basic.json",
    "mondo": "https://purl.obolibrary.org/obo/mondo.json",
    "cl": "https://github.com/obophenotype/cell-ontology/releases/latest/download/cl-base.json",
    "uberon": "https://purl.obolibrary.org/obo/uberon.json",
    "pato": "https://purl.obolibrary.org/obo/pato.json",
}

ONTOLOGIES_URL = [v for _, v in ONTOLOGIES_AS_DOC_MAP.items()]

LLM_CHANGE_AGENT_MODULE = pystow.module("llm_change_agent")
VECTOR_STORE = LLM_CHANGE_AGENT_MODULE.join("vector_store")
VECTOR_DB_PATH = VECTOR_STORE / "chroma.sqlite3"


PULL_REQUESTS_KEY = "pull_requests"
PR_CLOSED_ISSUES_KEY = "pr_closed_issues"
PR_COMMENTS_KEY = "pr_comments"
PR_CLOSED_ISSUE_COMMENT_KEY = "pr_closed_issue_comment"
PR_CLOSED_ISSUE_BODY_KEY = "issue_body"
PR_CLOSED_ISSUE_TITLE_KEY = "issue_title"
ID_KEY = "id"
CHANGES_KEY = "changes"

EVALUATION_PRS_FILE = "evaluation_prs.yaml"
