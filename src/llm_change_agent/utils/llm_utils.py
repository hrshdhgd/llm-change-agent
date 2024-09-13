"""Utility functions for the LLM Change Agent."""

import json
import logging
import re
from pathlib import Path
from typing import List, Union

import curies
import requests
import yaml
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from openai import OpenAI

from llm_change_agent.config.llm_config import AnthropicConfig, CBORGConfig, LLMConfig, OllamaConfig, OpenAIConfig
from llm_change_agent.constants import (
    ANTHROPIC_KEY,
    ANTHROPIC_PROVIDER,
    CBORG_KEY,
    CBORG_PROVIDER,
    CHROMA_MAX_BATCH_SIZE,
    KGCL_GRAMMAR,
    KGCL_SCHEMA,
    OLLAMA_PROVIDER,
    ONTOLOGIES_AS_DOC_MAP,
    OPENAI_KEY,
    OPENAI_PROVIDER,
    VECTOR_DB_NAME,
    VECTOR_DB_PATH,
    VECTOR_STORE,
)
from llm_change_agent.templates.templates import get_issue_analyzer_template, grammar_explanation

logger = logging.getLogger(__name__)


def get_openai_models():
    """Get the list of OpenAI models."""
    sorted_model_ids = []
    if OPENAI_KEY != "None":
        openai = OpenAI()
        models_list = sorted(
            [model for model in openai.models.list() if model.created >= 1706037777],
            key=lambda x: x.created,
            reverse=True,
        )
        sorted_model_ids = [model.id for model in models_list]
    return sorted_model_ids


def get_anthropic_models():
    """Get the list of Anthropic models."""
    return [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]


def get_ollama_models():
    """Get the list of Ollama models."""
    return ["llama3.1"]


def get_lbl_cborg_models():
    """Get the list of LBNL-hosted models via CBORG."""
    return [
        "lbl/cborg-chat:latest",  # LBL-hosted model
        "lbl/cborg-chat-nano:latest",  # LBL-hosted model
        "lbl/cborg-coder:latest",  # LBL-hosted model
        "openai/chatgpt:latest",  # LBL cloud-hosted model
        "anthropic/claude:latest",  # LBL cloud-hosted model
        "google/gemini:latest",  # LBL cloud-hosted model
    ]


def get_provider_model_map():
    """Get the provider to model mapping."""
    return {
        OPENAI_PROVIDER: get_openai_models(),
        OLLAMA_PROVIDER: get_ollama_models(),
        ANTHROPIC_PROVIDER: get_anthropic_models(),
        CBORG_PROVIDER: get_lbl_cborg_models(),
    }


def get_provider_for_model(model):
    """Get the provider for the model."""
    provider_model_map = get_provider_model_map()
    for provider, models in provider_model_map.items():
        if model in models:
            return provider
    return None


def get_default_model_for_provider(provider):
    """Get the default model for the provider."""
    provider_model_map = get_provider_model_map()
    if provider in provider_model_map:
        return provider_model_map[provider][0]
    return None


def get_api_key(provider):
    """Get the API key for the provider."""
    if provider == OPENAI_PROVIDER:
        return OPENAI_KEY
    elif provider == ANTHROPIC_PROVIDER:
        return ANTHROPIC_KEY
    elif provider == CBORG_PROVIDER:
        return CBORG_KEY
    return None


def llm_factory(config: LLMConfig):
    """Create an LLM instance based on the configuration."""
    if isinstance(config, OpenAIConfig):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.model, temperature=config.temperature, api_key=get_api_key(config.provider))
    elif isinstance(config, OllamaConfig):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.model, temperature=config.temperature, api_key=get_api_key(config.provider))
    elif isinstance(config, AnthropicConfig):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            api_key=get_api_key(config.provider),
            max_tokens_to_sample=4096,
        )
    elif isinstance(config, CBORGConfig):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            openai_api_key=get_api_key(config.provider),
            openai_api_base=config.base_url,
        )

    else:
        raise ValueError("Unsupported LLM configuration")


def get_kgcl_schema():
    """Get the KGCL schema information."""
    with open(KGCL_SCHEMA.locate(), "r") as schema_yaml:
        schema = yaml.safe_load(schema_yaml)
    return schema


def get_local_files_as_documents(path):
    """Get the local documents."""
    path_obj = Path(path)
    if path_obj.is_file():
        with open(path, "r") as doc_file:
            print(f"Reading from file: {path}")
            yield (Document(page_content=doc_file.read()),)
    elif path_obj.is_dir():
        for file_path in path_obj.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                with open(file_path, "r") as doc_file:
                    print(f"Reading from file: {file_path}")
                    yield Document(page_content=doc_file.read())
    else:
        return []


def get_kgcl_grammar():
    """Get the KGCL grammar information."""
    with open(KGCL_GRAMMAR.locate(), "r") as grammar_file:
        lark_file = grammar_file.read()
    grammar_notes = grammar_explanation()
    return {"lark": lark_file, "explanation": grammar_notes}


# def get_diff_docs():
#     """Download the diff docs."""
#     for url in ONTODIFF_DOCS:
#         # Extract the document name from the URL
#         doc_name = url.split("/")[-2]
#         doc_path = RAG_DOCS_DIR / f"{doc_name}.yaml"

#         # Check if the file already exists
#         if not doc_path.exists():
#             try:
#                 # Download the content from the URL
#                 response = requests.get(url, timeout=10)
#                 response.raise_for_status()  # Raise an error for bad status codes

#                 # Write the content to the file
#                 with open(doc_path, "w") as doc_file:
#                     doc_file.write(response.text)

#                 print(f"Downloaded and saved: {doc_name}")
#                 yield response.text

#             except requests.RequestException as e:
#                 print(f"Failed to download {url}: {e}")
#         else:
#             with open(doc_path, "r") as doc_file:
#                 print(f"Reading from file: {doc_name}")
#                 yield doc_file.read()


def get_vectorstore_path(docs, all_ontologies: bool = False):
    """Get the vector path."""
    if all_ontologies:
        return VECTOR_STORE / "all_ontologies" / VECTOR_DB_NAME

    if not docs:
        return VECTOR_DB_PATH

    doc_collection_name_list = []
    for doc in docs:
        if doc in ONTOLOGIES_AS_DOC_MAP.values():
            doc_collection_name_list.append(next(k for k, v in ONTOLOGIES_AS_DOC_MAP.items() if v == doc))
        elif Path(doc).exists():
            doc_collection_name_list.append(Path(doc).stem)
        elif doc.startswith("http"):
            doc_collection_name_list.append(doc.split("/")[-1].split(".")[0])
        else:
            doc_collection_name_list.append(doc)

    if doc_collection_name_list:
        return VECTOR_STORE / "_".join(doc_collection_name_list) / VECTOR_DB_NAME

    return VECTOR_DB_PATH


def split_documents(document: Union[str, Document], type: str = None):
    """Split the document into a list of documents."""
    if type == "json":
        splitter = RecursiveJsonSplitter(max_chunk_size=2000)  # default:2000
        splits_as_dicts = splitter.split_json(json_data=document, convert_lists=True)
        splits = [Document(page_content=json.dumps(split)) for split in splits_as_dicts]
    else:
        if isinstance(document, Document):
            doc_object = (document,)
        else:
            doc_object = (Document(page_content=document),)

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300, chunk_overlap=50)
        splits = splitter.split_documents(doc_object)

    return splits


def execute_agent(llm, prompt, external_rag_docs=None):
    """Create a retriever agent."""
    logger.info("Starting execution of the agent.")

    # Retrieve grammar and split documents
    grammar = get_kgcl_grammar()
    logger.info("Grammar retrieved successfully.")
    grammar_docs_list = split_documents(grammar["lark"]) + split_documents(grammar["explanation"])
    logger.info("Grammar documents split successfully.")

    ext_docs_list, ont_docs_list = [], []

    if external_rag_docs:
        logger.info("External documents provided. Loading external documents.")

        # Separate ontology documents from other external documents
        onts_in_docs = [doc for doc in external_rag_docs if doc in ONTOLOGIES_AS_DOC_MAP.values()]
        remaining_docs = list(set(external_rag_docs) - set(onts_in_docs))

        # Load and split ontology documents
        if onts_in_docs:
            list_of_ont_doc_lists = [requests.get(url, timeout=10).json() for url in onts_in_docs]
            ont_docs_list = [
                doc for docs in list_of_ont_doc_lists for doc in split_documents(document=docs, type="json")
            ]

        # Load and split remaining external documents
        if remaining_docs:
            list_of_ext_url_doc_lists = [
                WebBaseLoader(url, show_progress=True).load() for url in remaining_docs if url.startswith("http")
            ]
            list_of_ext_local_doc_lists = [
                get_local_files_as_documents(path)
                for path in remaining_docs
                if not path.startswith("http") and Path(path).exists()
            ]
            list_of_ext_doc_lists = list_of_ext_url_doc_lists + list_of_ext_local_doc_lists
            ext_docs_list = [
                split_doc for docs in list_of_ext_doc_lists for doc in docs for split_doc in split_documents(doc)
            ]

        logger.info("External documents loaded and split successfully.")

    # Combine all document lists
    docs_list = grammar_docs_list + ext_docs_list + ont_docs_list
    logger.info("All documents combined into a single list.")

    # Check for existing vector database path
    vector_db_path = get_vectorstore_path(external_rag_docs)
    embedding_model = OpenAIEmbeddings(show_progress_bar=True)

    if vector_db_path.exists():
        logger.info("Vector database path exists. Loading vectorstore from Chroma.")
        vectorstore = Chroma(embedding_function=embedding_model, persist_directory=str(vector_db_path.parent))
    else:
        logger.info("Vector database path does not exist. Creating new vectorstore.")

        # Function to split list into chunks
        def _chunk_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i : i + chunk_size]

        for batch in _chunk_list(docs_list, CHROMA_MAX_BATCH_SIZE):
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embedding_model, persist_directory=str(vector_db_path.parent)
            )

    logger.info("Vectorstore created from documents.")

    # Create retriever and tools
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    tool = create_retriever_tool(retriever, "change_agent_retriever", "Change Agent Retriever")
    tools = [tool, compress_iri]
    template = get_issue_analyzer_template()
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=template)
    agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        return_intermediate_steps=True,
    )
    logger.info("Agent executor created successfully.")
    return agent_executor.invoke(
        {
            "input": prompt,
            "grammar": grammar["lark"],
            "explanation": grammar["explanation"],
        }
    )


def extract_commands(command):
    """Extract the command from the list."""
    # Remove markdown markers
    cleaned_command = re.sub(r"```python|```", "", command).strip()

    # Define the regex pattern to match a list within square brackets
    pattern = r"\[.*?\]"

    # Search for the pattern in the cleaned input string
    match = re.search(pattern, cleaned_command, re.DOTALL)

    # If a match is found, return the matched string; otherwise, return the original cleaned command
    if match:
        return match.group(0)
    else:
        return cleaned_command


def normalize_to_curies_in_changes(changes: List):
    """Convert IRIs to CURIEs in change statements."""
    for idx, change in enumerate(changes):
        if any(string.startswith("<http") or string.startswith("http") for string in change.split()):
            iri = [string for string in change.split() if string.startswith("<http") or string.startswith("http")]
            # Replace the strings in the list with the curie using converter.compress(item)
            for _, item in enumerate(iri):
                stripped_item = item.strip("<>")
                compressed_item = compress_iri(stripped_item) if compress_iri(stripped_item) else item
                # Update the original change list with the compressed item
                change = change.replace(item, compressed_item)
                changes[idx] = change
    return changes


@tool
def compress_iri(iri: str) -> str:
    """Compress the IRI."""
    converter = curies.get_obo_converter()
    return converter.compress(iri)
