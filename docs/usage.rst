.. _usage:

Usage
=====

The CLI provides several commands to interact with the language models.

List Available Models
----------------------

To list all available models from supported providers:

.. code-block:: bash

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
      lbl/llama-3
      openai/chatgpt:latest
      anthropic/claude:latest
      google/gemini:latest

Generate Text
-------------

To generate text using a specified model and provider:

.. code-block:: bash

    llm-change execute --model <MODEL_NAME> --provider <PROVIDER_NAME> --prompt "<YOUR_PROMPT>"

Replace ``<MODEL_NAME>``, ``<PROVIDER_NAME>``, and ``<YOUR_PROMPT>`` with your desired values.

Examples
--------

Generate text using CBORG's ``lbl/llama-3`` model with a custom prompt:

.. code-block:: bash

    llm-change execute --model lbl/llama-3 --prompt "I want to change the definition of class ABC:123 to 'foo bar' and also create a new class labelled 'bar foo' with the curie DEF:123."

OR

.. code-block:: bash

    llm-change execute --provider cborg --prompt "I want to change the definition of class ABC:123 to 'foo bar' and also create a new class labelled 'bar foo' with the curie DEF:123."

generates

.. code-block:: bash

    Final Answer: 

    [
      "change definition of ABC:123 to 'foo bar'",
      "create class DEF:123 'bar foo'"
    ]

Evaluations
-----------

Input
~~~~~

The project also contains input data for evaluations in the form of YAML files for the following ontologies:

- `Envo <src/llm_change_agent/evaluations/input/EnvironmentOntology_envo.yaml>`_
- `GO <src/llm_change_agent/evaluations/input/geneontology_go_ontology.yaml>`_
- `MONDO <src/llm_change_agent/evaluations/input/monarch_initiative_mondo.yaml>`_
- `Cell Ontology <src/llm_change_agent/evaluations/input/obophenotype_cell_ontology.yaml>`_
- `Uberon <src/llm_change_agent/evaluations/input/obophenotype_uberon.yaml>`_
- `PATO <src/llm_change_agent/evaluations/input/pato_ontology_pato.yaml>`_

Expected changes
~~~~~~~~~~~~~~~~

The expected changes for specific pull requests are listed ontologywise below:

- `Envo <src/llm_change_agent/evaluations/input/expected/EnvironmentOntology_envo.yaml>`_
- `GO <src/llm_change_agent/evaluations/input/expected/geneontology_go_ontology.yaml>`_
- `MONDO <src/llm_change_agent/evaluations/input/expected/monarch_initiative_mondo.yaml>`_
- `Cell Ontology <src/llm_change_agent/evaluations/input/expected/obophenotype_cell_ontology.yaml>`_
- `Uberon <src/llm_change_agent/evaluations/input/expected/obophenotype_uberon.yaml>`_
- `PATO <src/llm_change_agent/evaluations/input/expected/pato_ontology_pato.yaml>`_

Actual results
~~~~~~~~~~~~~~

The actual results for some of the LLM models evaluated can be found `here <src/llm_change_agent/evaluations/output/>`_ and corresponding metrics `here <src/llm_change_agent/evaluations/output/metrics.yaml>`_

Development
-----------

To run the project locally, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/yourusername/llm-change-agent.git
    cd llm-change-agent
    poetry install

Make sure you have ``poetry`` installed in your system.