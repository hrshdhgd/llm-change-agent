"""Main python file."""

from pprint import pprint
from typing import List, Union

from llm_change_agent.constants import (
    ANTHROPIC_PROVIDER,
    CBORG_PROVIDER,
    OLLAMA_PROVIDER,
    OPEN_AI_MODEL,
    OPENAI_PROVIDER,
)
from llm_change_agent.utils.llm_utils import (
    augment_prompt,
    execute_agent,
    get_anthropic_models,
    get_default_model_for_provider,
    get_lbl_cborg_models,
    get_ollama_models,
    get_openai_models,
    get_provider_for_model,
    get_provider_model_map,
    llm_factory,
)


class LLMChangeAgent:
    """Define LLMChangeAgent class."""

    def __init__(self, model: str, prompt: str, provider: str, docs: Union[List, str]):
        """Initialize LLMChangeAgent class."""
        self.model = model
        self.prompt = prompt
        self.provider = provider
        self.llm = None
        self.docs = docs

    def _get_llm_config(self):
        """Get the LLM configuration based on the selected LLM."""

        def _validate_and_get_model(llm_model, get_models_func, default_model=OPEN_AI_MODEL):
            """Validate the model and get the model."""
            if llm_model is None:
                if self.provider is None:
                    from .config.llm_config import OpenAIConfig

                    return OpenAIConfig(model=default_model, provider=get_provider_for_model(default_model))
                else:
                    llm_model = get_default_model_for_provider(self.provider)

            list_of_models = get_models_func()
            if llm_model not in list_of_models:
                raise ValueError(f"Model {llm_model} not supported. Please choose from {list_of_models}")
            return llm_model

        provider_config_map = {
            OPENAI_PROVIDER: ("OpenAIConfig", get_openai_models),
            OLLAMA_PROVIDER: ("OllamaConfig", get_ollama_models),
            ANTHROPIC_PROVIDER: ("AnthropicConfig", get_anthropic_models),
            CBORG_PROVIDER: ("CBORGConfig", get_lbl_cborg_models),
        }

        if self.provider in provider_config_map:
            config_class_name, get_models_func = provider_config_map[self.provider]
            llm_model = _validate_and_get_model(self.model, get_models_func)
            config_module = __import__("llm_change_agent.config.llm_config", fromlist=[config_class_name])
            ConfigClass = getattr(config_module, config_class_name)
            return ConfigClass(model=llm_model, provider=get_provider_for_model(llm_model))

        else:
            all_models = get_provider_model_map()
            if self.model is None:
                from .config.llm_config import OpenAIConfig

                return OpenAIConfig(model=OPEN_AI_MODEL, provider=get_provider_for_model(OPEN_AI_MODEL))

            for provider, models in all_models.items():
                if self.model in models:
                    # Temporarily set the provider and model to use the existing logic
                    original_provider = self.provider
                    original_model = self.model
                    self.provider = provider
                    self.model = self.model
                    try:
                        return self._get_llm_config()
                    finally:
                        # Restore the original provider and model
                        self.provider = original_provider
                        self.model = original_model

            raise ValueError(f"Model {self.model} not supported.")

    def run(self):
        """Run the LLM Change Agent."""
        llm_config = self._get_llm_config()
        self.llm = llm_factory(llm_config)
        response = execute_agent(llm=self.llm, prompt=augment_prompt(self.prompt), docs=self.docs)
        pprint(response["output"])
        return response["output"]
