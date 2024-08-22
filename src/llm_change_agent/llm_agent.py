"""Main python file."""

from pprint import pprint

from llm_change_agent.constants import OPEN_AI_MODEL
from llm_change_agent.utils.llm_utils import (
    execute_agent,
    get_anthropic_models,
    get_lbl_cborg_models,
    get_ollama_models,
    get_openai_models,
    get_provider_for_model,
    get_provider_model_map,
    llm_factory,
)


class LLMChangeAgent:
    """Define LLMChangeAgent class."""

    def __init__(self, model: str, prompt: str, provider: str):
        """Initialize LLMChangeAgent class."""
        self.model = model
        self.prompt = prompt
        self.provider = provider
        self.llm = None

    def _get_llm_config(self):
        """Get the LLM configuration based on the selected LLM."""

        def _validate_and_get_model(llm_model, get_models_func, default_model=OPEN_AI_MODEL):
            """Validate the model and get the model."""
            if llm_model is None:
                return OpenAIConfig(model=default_model, provider=get_provider_for_model(llm_model))
            list_of_models = get_models_func()
            if llm_model not in list_of_models:
                raise ValueError(f"Model {llm_model} not supported. Please choose from {list_of_models}")
            return llm_model

        if self.provider == "openai":
            from .config.llm_config import OpenAIConfig

            llm_model = _validate_and_get_model(self.model, get_openai_models)
            return OpenAIConfig(model=llm_model, provider=get_provider_for_model(llm_model))

        elif self.provider == "ollama":
            from .config.llm_config import OllamaConfig

            llm_model = _validate_and_get_model(self.model, get_ollama_models)
            return OllamaConfig(model=llm_model)

        elif self.provider == "anthropic":
            from .config.llm_config import AnthropicConfig

            llm_model = _validate_and_get_model(self.model, get_anthropic_models)
            return AnthropicConfig(model=llm_model, provider=get_provider_for_model(llm_model))

        elif self.provider == "cborg":
            from .config.llm_config import CBORGConfig

            llm_model = _validate_and_get_model(self.model, get_lbl_cborg_models)
            return CBORGConfig(model=llm_model, provider=get_provider_for_model(llm_model))

        else:
            all_models = get_provider_model_map()
            if llm_model is None:
                llm_model = OPEN_AI_MODEL
                from .config.llm_config import OpenAIConfig

                return OpenAIConfig(model=llm_model, provider=get_provider_for_model(llm_model))

            for provider, models in all_models.items():
                if llm_model in models:
                    return self._get_llm_config(provider, llm_model)

            raise ValueError(f"Model {llm_model} not supported.")

    def run(self):
        """Run the LLM Change Agent."""
        llm_config = self._get_llm_config()
        self.prompt = """
        Add a new exact synonym gastric cancer for MONDO_0001056.
        """
        new_prompt = "Give me all relevant KGCL commands based on this request: " + self.prompt
        self.llm = llm_factory(llm_config)
        response = execute_agent(llm=self.llm, prompt=new_prompt)
        pprint(response["output"])
        return response["output"]
