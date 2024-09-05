.. _introduction:

Introduction
============

LLM Change Agent
----------------

.. image:: https://zenodo.org/badge/841604583.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.13693477

Overview
--------

LLM Change Agent is a command-line tool designed to interact with various large language models from different providers. It generates `KGCL commands <https://github.com/INCATools/kgcl/blob/main/src/data/examples/examples.yaml>`_ using specified models and providers via prompts. Prompts could be GitHub issue description/comments.

Features
--------

- Given a prompt relevant to making ontology resource changes, the agent responds with KGCL change statements.
- Supports OpenAI, Ollama, Anthropic and CBORG (LBNL hosted) models.

.. warning::

   OpenAI, Anthropic and CBORG model use are subject to availability of corresponding keys as environment variables locally.
