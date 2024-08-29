import click
import re

from llm_change_agent.constants import ONTOLOGIES_AS_DOC_MAP

def validate_path_or_url_or_ontology(ctx, param, value):
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    validated_values = []
    for val in value:
        if val.lower() in ONTOLOGIES_AS_DOC_MAP:
            validated_values.append(ONTOLOGIES_AS_DOC_MAP[val.lower()])
        elif re.match(url_pattern, val):
            validated_values.append(val)
        else:
            try:
                validated_values.append(click.Path(exists=True).convert(val, param, ctx))
            except click.BadParameter:
                raise click.BadParameter(f"{val} is not a valid URL, file path, or ontology name")
    
    return validated_values