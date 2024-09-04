.PHONY: all gpt-4o-2024-08-06 gemini-pro claude-sonnet

# Default target to run all evaluations
all: gpt-4o gemini-pro claude-sonnet

# Target for evaluating with gpt-4o-2024-08-06 model
gpt-4o:
	time llm-change evaluate --model gpt-4o-2024-08-06

# Target for evaluating with google/gemini-pro model
gemini-pro:
	time llm-change evaluate --model google/gemini-pro

# Target for evaluating with anthropic/claude-sonnet model
claude-sonnet:
	time llm-change evaluate --model anthropic/claude-sonnet
