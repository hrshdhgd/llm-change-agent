"""Templates for the LLM Change Agent."""

from langchain_core.prompts.prompt import PromptTemplate


def get_issue_analyzer_template():
    """Issue analyzer template."""
    template = """
        {input}

        You are an semantic engineer and an expert in Knowledge Graph Change Language (KGCL).
        Based on the GitHub issues you are given, you will form a list of relevant KGCL commands.
        You have the following tools at your disposal to help you with this task:
        {tools}
        You also have the KGCL grammar in lark format: {grammar} along with an explanation of the grammar: {explanation}.
        You MUST use CURIEs for every entity and relationship. You've been provided with JSON documents to find CURIEs/IRIs
        for entities and relationships. Do not manufacture CURIEs/IRIs. Make sure it is retrieved from these
        documents if absent in the GitHub issues provided. If you end up with a IRI to represent an entity, use
        the tool 'compress_iri' from {tools} to derive a CURIE from it. If you end up with the label
        for the entity, try to retrieve its CURIE/IRI from the JSON docs and get CURIE using {tools}.

        For e.g.: if you have a change `delete edge MONDO:0005772 rdfs:subClassOf <opportunistic mycosis>`
        It should be converted to `delete edge MONDO:0005772 rdfs:subClassOf MONDO:0002312` based on RAG.
          
        The final answer should be JUST a list of KGCL commands, nothing else.
        Keep the verbosity of the response to zero. It should be concise and to the point.
        Do not truncate the commands. Write it out completely as per the grammar.

        It is fine if you are not able to form any commands. You can just return an empty list.

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
