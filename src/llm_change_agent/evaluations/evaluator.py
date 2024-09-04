"""Evaluation script for the LLM Change Agent."""

import ast
import logging
import os
import random
import secrets
import time
from pathlib import Path
from typing import Any, List, Union

import click
import requests
import yaml

from llm_change_agent.constants import (
    CHANGES_KEY,
    EVALUATION_PRS_FILE,
    ID_KEY,
    ONTODIFF_DOCS,
    PR_CLOSED_ISSUE_BODY_KEY,
    PR_CLOSED_ISSUE_COMMENT_KEY,
    PR_CLOSED_ISSUE_TITLE_KEY,
    PR_CLOSED_ISSUES_KEY,
    PULL_REQUESTS_KEY,
)
from llm_change_agent.utils.llm_utils import extract_commands, normalize_to_curies_in_changes

logger = logging.getLogger(__name__)
logger.info("Evaluating the LLM Change Agent.")


def download_document(url, input_dir):
    """Download the document from the URL."""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    # Extract the document name from the URL
    doc_name = url.split("/")[-2].replace("-", "_") + ".yaml"
    file_path = Path(input_dir) / doc_name
    if file_path.exists():
        print(f"File {doc_name} already exists in {input_dir}")
        return
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded {doc_name} to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


def prepare_eval_and_expected_yamls(input_dir: Union[str, Path]):
    """Prepare the evaluation and expected YAMLs for the input documents."""
    input_dir = Path(input_dir)
    eval_dir = input_dir / "eval_docs"
    expected_dir = input_dir / "expected"

    eval_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    for doc in input_dir.iterdir():
        if doc.is_file() and doc.suffix == ".yaml":
            new_doc = doc.name
            if Path(expected_dir / new_doc).exists() and Path(eval_dir / new_doc).exists():
                print(f"Skipping {new_doc} as it already exists in both the evaluation and expected datasets.")
                continue
            with open(doc, "r") as ex:
                doc_yaml = yaml.safe_load(ex)

            prs = doc_yaml.get(PULL_REQUESTS_KEY, [])
            for idx, pr in enumerate(prs):
                pr_id = pr.get(ID_KEY)
                pr_changes = pr.get(CHANGES_KEY)
                pr_closed_issues = pr.get(PR_CLOSED_ISSUES_KEY, [])
                all_closed_issues = []

                for issue in pr_closed_issues:
                    pr_closed_issue_title = issue.get(PR_CLOSED_ISSUE_TITLE_KEY, "")
                    pr_closed_issue_body = issue.get(PR_CLOSED_ISSUE_BODY_KEY, "")
                    pr_closed_issue_comments = issue.get(PR_CLOSED_ISSUE_COMMENT_KEY, "")

                    # Filter out None and empty strings, then split into lines and filter out empty lines
                    issue_prompt_parts = [pr_closed_issue_title, pr_closed_issue_body, pr_closed_issue_comments]
                    issue_prompt = "\n".join(
                        line for part in issue_prompt_parts if part for line in part.splitlines() if line.strip()
                    )

                    prompt_block = {pr_id: issue_prompt}
                    all_closed_issues.append(prompt_block)

                eval_block = {pr_id: pr_changes}

                if idx == 0:
                    mode = "w"
                else:
                    mode = "a"

                with (expected_dir / new_doc).open(mode) as ex, (eval_dir / new_doc).open(mode) as ev:
                    yaml.dump(eval_block, ex, sort_keys=False, default_flow_style=False)
                    yaml.dump(all_closed_issues, ev, sort_keys=False, default_flow_style=False)
    return eval_dir, expected_dir


def run_llm_change_agent(prompt, provider, model, docs: List[Any] = None) -> List:
    """Run the LLM Change Agent."""
    if not docs:
        docs = []
    from llm_change_agent.cli import execute

    # Sleep for a random time between 1 and 5 seconds before running the LLM Change Agent
    sleep_time = secrets.randbelow(5) + 1
    logger.info(f"Sleeping for {sleep_time} seconds before running the LLM Change Agent.")
    time.sleep(sleep_time)

    with click.Context(execute) as ctx:
        ctx.params["prompt"] = prompt
        ctx.params["provider"] = provider
        ctx.params["model"] = model
        ctx.params["docs"] = docs
        response = extract_commands(execute.invoke(ctx))
        kgcl_commands = [command for command in ast.literal_eval(response)]
        return kgcl_commands


def generate_changes_via_llm(eval_dir, output_dir, provider, model):
    """Generate changes via the LLM Change Agent."""
    os.makedirs(output_dir, exist_ok=True)
    output_sub_dir = output_dir / provider / model.split(":")[0].replace("/", "_")
    os.makedirs(output_sub_dir, exist_ok=True)
    pr_eval_list_filepath = Path(eval_dir).parent / EVALUATION_PRS_FILE
    if pr_eval_list_filepath.exists():
        with open(pr_eval_list_filepath, "r") as f:
            doc_pr_dict = yaml.safe_load(f)
    else:
        doc_pr_dict = {}

    for doc in eval_dir.iterdir():
        update_eval_dict = False
        if doc.is_file() and doc.suffix == ".yaml":
            if (output_sub_dir / doc.name).exists():
                print(f"Skipping {doc.name} as it already exists in the output directory.")
                continue
            with open(doc, "r") as f:
                eval_yaml_list = yaml.safe_load(f)
            if doc_pr_dict.get(doc.name):
                update_eval_dict = False
                pr_eval_list = doc_pr_dict.get(doc.name)
                sampled_evals = [
                    {k: v} for eval_yaml in eval_yaml_list for k, v in eval_yaml.items() if k in pr_eval_list
                ]
                sample_size = len(sampled_evals)
            else:
                doc_pr_dict[doc.name] = []
                update_eval_dict = True
                sample_size = max(10, len(eval_yaml_list) // 200)
                sampled_evals = random.sample(eval_yaml_list, sample_size)
            logger.info(f"Running evaluation on {sample_size} pull request related issues for {doc.name}")
            for idx, combo in enumerate(sampled_evals):
                if idx == 0:
                    mode = "w"
                else:
                    mode = "a"
                pr_id, issue = next(iter(combo.items()))
                if update_eval_dict:
                    doc_pr_dict[doc.name].append(pr_id)
                    with open(pr_eval_list_filepath, "w") as f:
                        yaml.dump(doc_pr_dict, f, sort_keys=False, default_flow_style=False)

                prompt = issue
                try:
                    predicted_changes = run_llm_change_agent(prompt, provider, model)
                except Exception as e:
                    logger.error(f"Error while generating changes for {doc.name} and PR {pr_id}: {e}")
                    predicted_changes = []

                with open(output_sub_dir / doc.name, mode) as out:
                    yaml.dump({pr_id: predicted_changes}, out, sort_keys=False)

    print(f"Predicted changes saved to {output_sub_dir}")


def compare_changes(expected_dir: Path, output_dir: Path):
    """Compare the actual changes with the predicted changes."""
    # For each document in the expected directory, there is a corresponding document in the output directory

    output_files = list(output_dir.rglob("*.yaml"))

    # output_files_dict is : {provider_model: {filename: file_path}}
    output_files_list_of_dicts = [{f"{file.parts[-3]}_{file.parts[-2]}": {file.name: file}} for file in output_files]
    jaccard_score_dict = {}
    for model_output in output_files_list_of_dicts:
        for provider_model, file_info in model_output.items():
            jaccard_score_dict[provider_model] = {}
            for filename, filepath in file_info.items():
                filename = filepath.name
                expected_file = expected_dir / filename
                output_file = filepath
                with open(expected_file, "r") as ex, open(output_file, "r") as out:
                    expected_yaml = yaml.safe_load(ex)
                    output_yaml = yaml.safe_load(out)
                expected_yaml_subset = {k: v for k, v in expected_yaml.items() if k in output_yaml}
                for pr_id, output_changes in output_yaml.items():
                    expected_change = expected_yaml_subset.get(pr_id)
                    if len(output_changes) > 0:
                        jaccard_score_dict[provider_model][pr_id] = get_comparison_metrics(
                            expected_change, output_changes
                        )
        logger.info(f"Finished comparing changes for {provider_model}")
    with open(output_dir / "metrics.yaml", "a") as f:
        yaml.dump(jaccard_score_dict, f, sort_keys=False, default_flow_style=False)


def get_comparison_metrics(expected_changes: List, output_changes: List):
    """Compare the expected changes with the output changes."""
    output_changes = normalize_to_curies_in_changes(output_changes)
    expected_changes = normalize_to_curies_in_changes(expected_changes)
    # Calculate Jaccard between the expected and output changes
    expected_changes_set = set(expected_changes)
    output_changes_set = set(output_changes)
    intersection = expected_changes_set.intersection(output_changes_set)
    union = expected_changes_set.union(output_changes_set)
    jaccard = len(intersection) / len(union)
    logger.info(f"Jaccard similarity between expected and output changes: {jaccard}")
    #  Caclulate accuracy between the expected and output changes
    accuracy = len(intersection) / len(expected_changes_set)

    metrics = {
        "jaccard": jaccard,
        "accuracy": accuracy,
    }

    return metrics


def run_evaluate(model: str, provider: str):
    """Evaluate the LLM Change Agent."""
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    if (output_dir / "metrics.yaml").exists():
        # delete file
        os.remove(output_dir / "metrics.yaml")

    logger.info("Downloading the ONTODIFF_DOCS into the input directory.")

    for url in ONTODIFF_DOCS:
        logger.info(f"Downloading {url} into the input directory.")
        download_document(url, input_dir)

    eval_dir, expected_dir = prepare_eval_and_expected_yamls(input_dir)

    generate_changes_via_llm(model=model, provider=provider, eval_dir=eval_dir, output_dir=output_dir)

    compare_changes(expected_dir=expected_dir, output_dir=output_dir)
