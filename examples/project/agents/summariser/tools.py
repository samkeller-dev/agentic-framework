from pydantic import BaseModel


def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


class Output(BaseModel):
    summary: str
    word_count: int
