"""Evaluation script for the LLM Change Agent."""

import ast
import logging
import os
from pathlib import Path
import random
from typing import List, Union

import click
import requests
import yaml

from llm_change_agent.constants import (
    CHANGES_KEY,
    ID_KEY,
    ONTOLOGIES_AS_DOC_MAP,
    OPEN_AI_MODEL,
    OPENAI_PROVIDER,
    PR_CLOSED_ISSUE_BODY_KEY,
    PR_CLOSED_ISSUE_COMMENT_KEY,
    PR_CLOSED_ISSUE_TITLE_KEY,
    PR_CLOSED_ISSUES_KEY,
    PULL_REQUESTS_KEY,
)

logger = logging.getLogger(__name__)
logger.info("Evaluating the LLM Change Agent.")


def download_document(url, input_dir):
    """Download the document from the URL."""
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    # Extract the document name from the URL
    doc_name = url.split("/")[-2].replace("-", "_").lower().split("_")[-1] + ".yaml"
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


def prepare_evaland_expected_yamls(input_dir: Union[str, Path]):
    """Prepare the evaluation and expected YAMLs for the input documents."""
    input_dir = Path(input_dir)
    eval_dir = input_dir / "eval_docs"
    expected_dir = input_dir / "expected"

    eval_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    for idx, doc in enumerate(input_dir.iterdir()):
        if doc.is_file() and doc.suffix == ".yaml":
            new_doc = doc.name
            if Path(expected_dir / new_doc).exists() and Path(eval_dir / new_doc).exists():
                print(f"Skipping {new_doc} as it already exists in both the evaluation and expected datasets.")
                continue
            with open(doc, "r") as ex:
                doc_yaml = yaml.safe_load(ex)

            prs = doc_yaml.get(PULL_REQUESTS_KEY, [])
            for pr in prs:
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


def run_llm_change_agent(prompt, provider, model, docs=[]) -> List:
    from llm_change_agent.cli import execute

    with click.Context(execute) as ctx:
        ctx.params["prompt"] = prompt
        ctx.params["provider"] = provider
        ctx.params["model"] = model
        ctx.params["docs"] = docs
        response = execute.invoke(ctx)
        kgcl_commands = [
            command.replace('"', "'").replace("```python\n'", "").replace("'```')", "")
            for command in ast.literal_eval(response)
        ]
        return kgcl_commands


def run_evaluation_script(eval_dir, output_dir):
    """Evaluate the LLM Change Agent."""
    os.makedirs(output_dir, exist_ok=True)
    for doc in eval_dir.iterdir():
        if doc.is_file() and doc.suffix == ".yaml":
            with open(doc, "r") as f:
                eval_yaml = yaml.safe_load(f)
            sample_size = max(10, len(eval_yaml) // 100)
            sampled_evals = random.sample(eval_yaml, sample_size)
            logger.info(f"Running evaluation on {sample_size} pull request related issues for {doc.name}")
            for combo in sampled_evals:
                pr_id, issue = next(iter(combo.items()))
                prompt = issue
                provider = OPENAI_PROVIDER
                model = OPEN_AI_MODEL
                predicted_changes = run_llm_change_agent(prompt, provider, model, ONTOLOGIES_AS_DOC_MAP.get(doc.stem))

                with open(output_dir / doc.name, "a") as out:
                    yaml.dump({pr_id: predicted_changes}, out, sort_keys=False)
    print(f"Predicted changes saved to {output_dir}")


def compare_changes():
    # Placeholder function to simulate comparison of changes
    # In a real scenario, you would implement the logic to compare actual vs predicted changes
    print("Comparing actual changes with predicted changes")


def run_evaluate():
    """Evaluate the LLM Change Agent."""
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"

    logger.info("Downloading the ONTODIFF_DOCS into the input directory.")
    # !UNCOMMENT THIS
    # for url in ONTODIFF_DOCS:
    #     logger.info(f"Downloading {url} into the input directory.")
    #     download_document(url, input_dir)

    eval_dir, expected_dir = prepare_evaland_expected_yamls(input_dir)

    run_evaluation_script(eval_dir, output_dir)

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
