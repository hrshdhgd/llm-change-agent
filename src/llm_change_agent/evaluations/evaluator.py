"""Evaluation script for the LLM Change Agent."""

import ast
import logging
import os
import random
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
from llm_change_agent.utils.llm_utils import extract_commands

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
                predicted_changes = run_llm_change_agent(prompt, provider, model)

                with open(output_sub_dir / doc.name, mode) as out:
                    yaml.dump({pr_id: predicted_changes}, out, sort_keys=False)

    print(f"Predicted changes saved to {output_sub_dir}")


def compare_changes():
    """Compare the actual changes with the predicted changes."""
    import pdb

    pdb.set_trace()


def run_evaluate(model: str, provider: str):
    """Evaluate the LLM Change Agent."""
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"

    logger.info("Downloading the ONTODIFF_DOCS into the input directory.")

    for url in ONTODIFF_DOCS:
        logger.info(f"Downloading {url} into the input directory.")
        download_document(url, input_dir)

    eval_dir, expected_dir = prepare_eval_and_expected_yamls(input_dir)

    generate_changes_via_llm(model=model, provider=provider, eval_dir=eval_dir, output_dir=output_dir)

    # compare_changes()

    # logger.info("Split the YAML documents randomly into RAG and Evaluation documents 80% and 20%.")
    # random.shuffle(ONTODIFF_DOCS)
    # split_index = int(len(ONTODIFF_DOCS) * 0.8)
    # rag_docs = ONTODIFF_DOCS[:split_index]
    # eval_docs = ONTODIFF_DOCS[split_index:]

    # logger.info("Run llm_change_agent with the RAG documents.")
    # run_llm_change_agent(rag_docs)

    # logger.info("Run the evaluation script with the Evaluation documents.")
    # run_evaluation_script(eval_docs)

    # logger.info("Compare the actual `changes` with the predicted `changes` from the llm_change_agent.")
    # compare_changes()

    # logger.info("Evaluation completed.")
    # return
