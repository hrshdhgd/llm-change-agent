"""Utility functions for the LLM Change Agent."""

import yaml
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from llm_change_agent.config.llm_config import AnthropicConfig, CBORGConfig, LLMConfig, OllamaConfig, OpenAIConfig
from llm_change_agent.constants import ANTHROPIC_KEY, CBORG_KEY, KGCL_GRAMMAR, KGCL_SCHEMA, OPENAI_KEY
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


def split_documents(document: str):
    """Split the document into a list of documents."""
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

    docs_list = split_documents(grammar["lark"]) + split_documents(grammar["explanation"])
    vectorstore = Chroma.from_documents(documents=docs_list, embedding=OpenAIEmbeddings())
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
        \n\nReturn as a JSON format list.\n\n"""
