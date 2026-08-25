import tiktoken

# Groq-hosted models don't publish a public tokenizer; cl100k_base is used as
# an approximation for token counting/cost estimation, not an exact count.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))
