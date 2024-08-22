"""Templates for the LLM Change Agent."""

from langchain_core.prompts.prompt import PromptTemplate


def get_issue_analyzer_template():
    """Issue analyzer template."""
    template = """
        {input}
        You are an semantic engineer. Based on the text you are given,
        you will analyze it.
        
        All that is expected of you is to form relevant KGCL commands in a list format.
        You have the following tools at your disposal to help you with this task:
        {tools}
        You also have the KGCL grammar in lark format: {grammar} along with an explanation of the grammar: {explanation}
        I want NO verbosity in your response. The final answer should consistently be a list of commands.

        Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
    """
    return PromptTemplate(
        input_variables=[
            "input",
            "agent_scratchpad",
            "tools",
            "tool_names",
            "intermediate_steps",
            # "schema",
            "grammar",
            "explanation",
        ],
        template=template,
    )


def grammar_explanation():
    """Grammar explanation template."""
    return """
    The grammar defines commands for various operations such as renaming, creating, deleting, and modifying entities. It includes the following components:

    - `expression`: The entry point of the grammar, which can be any of the defined commands like `rename`, `create`, `delete`, etc.

    - `rename`: This command follows the pattern `"rename" _WS [id _WS "from" _WS] old_label ["@" old_language] _WS ("to"|"as"|"->") _WS new_label ["@" new_language]`.

    - `create`: This command follows the pattern `"create node" _WS id _WS label ["@" language]`.

    - `create_class`: This command follows the pattern `"create" _WS id`.

    - `create_synonym`: This command follows the pattern `"create" _WS [synonym_qualifier _WS] "synonym" _WS synonym ["@" language] _WS "for" _WS entity`.

    - `delete`: This command follows the pattern `"delete" _WS entity`.

    - `obsolete`: This command follows the pattern `"obsolete" _WS entity` or `"obsolete" _WS entity _WS "with replacement" _WS replacement`.

    - `unobsolete`: This command follows the pattern `"unobsolete" _WS entity`.

    - `deepen`: This command follows the pattern `"deepen" _WS entity _WS "from" _WS old_entity _WS ("to"|"->") _WS new_entity`.

    - `shallow`: This command follows the pattern `"shallow" _WS entity _WS "from" _WS old_entity _WS ("to"|"->") _WS new_entity`.

    - `move`: This command follows the pattern `"move" _WS entity_subject _WS entity_predicate _WS entity_object _WS "from" _WS old_entity _WS ("to"|"as"|"->") _WS new_entity`.

    - `create_edge`: This command follows the pattern `"create edge" _WS entity_subject _WS entity_predicate _WS entity_object_id`.

    - `delete_edge`: This command follows the pattern `"delete edge" _WS entity_subject _WS entity_predicate _WS entity_object_id`.

    - `change_relationship`: This command follows the pattern `"change relationship between" _WS entity_subject _WS "and" _WS entity_object _WS "from" _WS old_entity _WS "to" _WS new_entity`.

    - `change_annotation`: This command follows the pattern `"change annotation of" _WS entity_subject _WS "with" _WS entity_predicate _WS "from" _WS old_entity_object _WS "to" _WS new_entity_object`.

    - `change_definition`: This command follows the pattern `"change definition of" _WS entity _WS "to" _WS new_definition` or `"change definition of" _WS entity _WS "from" _WS old_definition _WS "to" _WS new_definition`.

    - `add_definition`: This command follows the pattern `"add definition" _WS new_definition _WS "to" _WS entity`.

    - `remove_definition`: This command follows the pattern `"remove definition for" _WS entity`.

    - `remove_from_subset`: This command follows the pattern `"remove" _WS id _WS "from subset" _WS subset`.

    - `add_to_subset`: This command follows the pattern `"add" _WS id _WS "to subset" _WS subset`.

    The `%import` statements import common definitions for `ID`, `LABEL`, `CURIE`, `SINGLE_QUOTE_LITERAL`, `TRIPLE_SINGLE_QUOTE_LITERAL`, `DOUBLE_QUOTE_LITERAL`, `TRIPLE_DOUBLE_QUOTE_LITERAL`, `LANGUAGE_TAG`, and whitespace (`_WS`). The `%ignore` statement tells the parser to ignore whitespace.

    """
