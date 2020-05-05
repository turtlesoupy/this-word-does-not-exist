import json
import random

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Word:
    word: str
    definition: str

    pos: Optional[str]
    topic: Optional[str]
    example: Optional[str]
    syllables: Optional[List[str]]

    @classmethod
    def from_dict(cls, d):
        return cls(
            word=d["word"],
            definition=d["definition"],
            pos=d["pos"] if "pos" in d else None,
            topic=d["topic"] if "topic" in d else None,
            example=d["example"] if "example" in d else None,
            syllables=d["syllables"] if "syllables" in d else None,
        )

    def to_dict(self):
        return {
            "word": self.word,
            "definition": self.definition,
            "pos": self.pos,
            "topic": self.topic,
            "example": self.example,
            "syllables": self.syllables,
        }


class WordIndex:
    def __init__(self, words):
        self.words = words
        self.word_index = {e.word: e for e in self.words}

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            words = [Word.from_dict(e) for e in json.load(f)]
        return cls(words)

    def dump(self, path):
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.words], f)

    def random(self):
        return random.choice(self.words)

    def by_name(self, name):
        return self.word_index[name]
