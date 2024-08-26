"""Utility functions for the LLM Change Agent."""

from typing import Union

import yaml
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from llm_change_agent.config.llm_config import AnthropicConfig, CBORGConfig, LLMConfig, OllamaConfig, OpenAIConfig
from llm_change_agent.constants import (
    ANTHROPIC_KEY,
    CBORG_KEY,
    KGCL_GRAMMAR,
    KGCL_SCHEMA,
    ONTODIFF_DOCS,
    OPENAI_KEY,
    VECTOR_DB_PATH,
    VECTOR_STORE,
)
from llm_change_agent.templates.templates import get_issue_analyzer_template, grammar_explanation


def get_openai_models():
    """Get the list of OpenAI models."""
    sorted_model_ids = []
    if OPENAI_KEY != "None":
        openai = OpenAI()
        models_list = sorted(
            [model for model in openai.models.list() if model.id.startswith("gpt-4") and model.created >= 1706037777],
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
        "lbl/llama-3",  # LBNL-hosted model (free to use)
        "openai/chatgpt:latest",  # OpenAI-hosted model
        "anthropic/claude:latest",  # Anthropic-hosted model
        "google/gemini:latest",  # Google-hosted model
    ]


def get_provider_model_map():
    """Get the provider to model mapping."""
    return {
        "openai": get_openai_models(),
        "ollama": get_ollama_models(),
        "anthropic": get_anthropic_models(),
        "cborg": get_lbl_cborg_models(),
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
    if provider == "openai":
        return OPENAI_KEY
    elif provider == "anthropic":
        return ANTHROPIC_KEY
    elif provider == "cborg":
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


def split_documents(document: Union[str, Document]):
    """Split the document into a list of documents."""
    if isinstance(document, Document):
        doc_object = (document,)
    else:
        doc_object = (Document(page_content=document),)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(doc_object)
    return splits


def execute_agent(llm, prompt):
    """Create a retriever agent."""
    grammar = get_kgcl_grammar()
    # schema = get_kgcl_schema()
    # docs_list = (
    #     split_documents(str(schema)) + split_documents(grammar["lark"]) + split_documents(grammar["explanation"])
    # )
    grammar_docs_list = split_documents(grammar["lark"]) + split_documents(grammar["explanation"])
    if VECTOR_DB_PATH.exists():
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(show_progress_bar=True), persist_directory=str(VECTOR_STORE)
        )
    else:

        list_of_doc_lists = [WebBaseLoader(url, show_progress=True).load() for url in ONTODIFF_DOCS]
        diff_docs_list = [split_doc for docs in list_of_doc_lists for doc in docs for split_doc in split_documents(doc)]
        docs_list = grammar_docs_list + diff_docs_list

        vectorstore = Chroma.from_documents(
            documents=docs_list, embedding=OpenAIEmbeddings(show_progress_bar=True), persist_directory=str(VECTOR_STORE)
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    tool = create_retriever_tool(retriever, "change_agent_retriever", "Change Agent Retriever")
    tools = [tool]
    template = get_issue_analyzer_template()
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=template)
    agent_executor = AgentExecutor(agent=react_agent, tools=tools, handle_parsing_errors=True, verbose=True)

    return agent_executor.invoke(
        {
            "input": prompt,
            # "schema": schema,
            "grammar": grammar["lark"],
            "explanation": grammar["explanation"],
        }
    )


def augment_prompt(prompt: str):
    """Augment the prompt with additional information."""
    return f"""
        Give me all relevant KGCL commands based on this request: \n\n
        + {prompt} +
        \n\n
        Return as a python list object which will be passed to another tool.
        Each element of the list should be enlosed in double quotes.
        """
