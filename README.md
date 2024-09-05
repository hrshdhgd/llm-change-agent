# LLM Change Agent
[![DOI](https://zenodo.org/badge/841604583.svg)](https://zenodo.org/doi/10.5281/zenodo.13693477)
## Overview
LLM Change Agent is a command-line tool designed to generate knowledge graph change language (KGCL) commands that request changes in a knowledge graph such as an ontology.
It generates [KGCL commands](https://github.com/INCATools/kgcl/blob/main/src/data/examples/examples.yaml) using specified large language models and providers via prompts.
The prompts can be GitHub issue description/comments or natural language text.

## Features
- Given a prompt relevant to making ontology resource changes, the agent responds with KGCL change statements.
- Supports OpenAI, Ollama, Anthropic and CBORG (LBNL-hosted) models.

**:warning:** OpenAI, Anthropic and CBORG model use require availability of corresponding keys as environment variables locally.

## Installation
To install the dependencies, run:
```bash
pip install llm-change-agent
```

## Usage
The CLI provides several commands to interact with the language models.

### List Available Models
To list all available models from supported providers:
```bash
llm-change list-models

OpenAI models:
  gpt-4o-2024-08-06
  gpt-4o-mini
  gpt-4o-mini-2024-07-18
  gpt-4o-2024-05-13
  gpt-4o
  gpt-4-turbo-2024-04-09
  gpt-4-turbo
  gpt-4-turbo-preview
Anthropic models:
  claude-3-5-sonnet-20240620
  claude-3-opus-20240229
  claude-3-sonnet-20240229
  claude-3-haiku-20240307
Ollama models:
  llama3.1
LBL-CBORG models:
  lbl/cborg-chat:latest
  lbl/cborg-chat-nano:latest
  openai/gpt-3.5-turbo
  openai/gpt-4o
  openai/gpt-4o-mini
  anthropic/claude-haiku
  anthropic/claude-sonnet
  anthropic/claude-opus
  google/gemini-pro
  google/gemini-flash
```

### Generate Text
To generate text using a specified model and provider:
```bash
llm-change execute --model <MODEL_NAME> --provider <PROVIDER_NAME> --prompt "<YOUR_PROMPT>"
```
Replace `<MODEL_NAME>`, `<PROVIDER_NAME>`, and `<YOUR_PROMPT>` with your desired values.

### Examples
- Generate text using CBORG's `lbl/llama-3` model with a custom prompt:
```bash
llm-change execute --model lbl/llama-3 --prompt "I want to change the definition of class ABC:123 to 'foo bar' and also create a new class labelled 'bar foo' with the curie DEF:123."
```
OR
```bash
llm-change execute --provider cborg --prompt "I want to change the definition of class ABC:123 to 'foo bar' and also create a new class labelled 'bar foo' with the curie DEF:123."
```

generates

```bash
Final Answer: 

[
  "change definition of ABC:123 to 'foo bar'",
  "create class DEF:123 'bar foo'"
]

```

## Evaluations
### Input
The project also contains input data for evaluations in the form of YAML files for the following ontologies:
- [Envo](src/llm_change_agent/evaluations/input/EnvironmentOntology_envo.yaml)
- [GO](src/llm_change_agent/evaluations/input/geneontology_go_ontology.yaml)
- [MONDO](src/llm_change_agent/evaluations/input/monarch_initiative_mondo.yaml)
- [Cell Ontology](src/llm_change_agent/evaluations/input/obophenotype_cell_ontology.yaml)
- [Uberon](src/llm_change_agent/evaluations/input/obophenotype_uberon.yaml)
- [PATO](src/llm_change_agent/evaluations/input/pato_ontology_pato.yaml)

### Expected changes
The expected changes for specific pull requests are listed ontologywise below:
- [Envo](src/llm_change_agent/evaluations/input/expected/EnvironmentOntology_envo.yaml)
- [GO](src/llm_change_agent/evaluations/input/expected/geneontology_go_ontology.yaml)
- [MONDO](src/llm_change_agent/evaluations/input/expected/monarch_initiative_mondo.yaml)
- [Cell Ontology](src/llm_change_agent/evaluations/input/expected/obophenotype_cell_ontology.yaml)
- [Uberon](src/llm_change_agent/evaluations/input/expected/obophenotype_uberon.yaml)
- [PATO](src/llm_change_agent/evaluations/input/expected/pato_ontology_pato.yaml)

### Actual results
The actual results for some of the LLM models evaluated are [here](src/llm_change_agent/evaluations/output/) and corresponding metrics [here](src/llm_change_agent/evaluations/output/metrics.yaml)

## Development
To run the project locally, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/llm-change-agent.git
cd llm-change-agent
poetry install
```
Make sure you have `poetry` installed in your system.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
### Acknowledgements

This [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/README.html) project was developed from the [monarch-project-template](https://github.com/monarch-initiative/monarch-project-template) template and will be kept up-to-date using [cruft](https://cruft.github.io/cruft/).
