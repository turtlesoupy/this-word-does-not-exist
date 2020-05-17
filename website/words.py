import json
import random
import gzip

from cryptography.fernet import Fernet
from dataclasses import dataclass
from typing import Optional, List
from word_service.word_service_proto import wordservice_pb2


@dataclass
class Word:
    word: str
    definition: str

    pos: Optional[str]
    topic: Optional[str]
    example: Optional[str]
    syllables: Optional[List[str]]
    probably_exists: Optional[bool]
    dataset_type: Optional[int]

    @classmethod
    def from_protobuf(cls, proto: wordservice_pb2.WordDefinition):
        example = None
        if proto.examples:
            example = proto.examples[0]

        return cls(
            word=proto.word,
            definition=proto.definition,
            pos=proto.pos,
            topic=None,
            example=example,
            syllables=list(proto.syllables),
            probably_exists=proto.probablyExists,
            dataset_type=(proto.dataset or None),
        )

    @classmethod
    def from_dict(cls, d):
        # Short dict
        if "w" in d and "d" in d:
            return cls(
                word=d["w"],
                definition=d["d"],
                pos=d["p"] if "p" in d else None,
                topic=d["t"] if "t" in d else None,
                example=d["e"] if "e" in d else None,
                syllables=d["s"] if "s" in d else None,
                probably_exists=d["l"] if "l" in d else None,
                dataset_type=d["dt"] if "dt" in d else None,
            )
        else:
            return cls(
                word=d["word"],
                definition=d["definition"],
                pos=d["pos"] if "pos" in d else None,
                topic=d["topic"] if "topic" in d else None,
                example=d["example"] if "example" in d else None,
                syllables=d["syllables"] if "syllables" in d and len(d["syllables"]) > 0 else None,
                probably_exists=d["probably_exists"] if "probably_exists" in d else None,
                dataset_type=d["dataset_type"] if "dataset_type" in d else None,
            )

    def to_short_dict(self):
        ret = {
            "w": self.word,
            "d": self.definition,
        }

        if self.pos:
            ret["p"] = self.pos
        if self.topic:
            ret["t"] = self.topic
        if self.example:
            ret["e"] = self.example
        if self.syllables and len(self.syllables) > 1:
            ret["s"] = self.syllables
        if self.probably_exists:
            ret["l"] = self.probably_exists
        if self.dataset_type:
            ret["dt"] = self.dataset_type

        return ret

    def to_dict(self):
        return {
            "word": self.word,
            "definition": self.definition,
            "pos": self.pos,
            "topic": self.topic,
            "example": self.example,
            "syllables": self.syllables,
            "probably_exists": self.probably_exists,
            "dataset_type": self.dataset_type,
        }


class WordIndex:
    def __init__(self, words):
        self.words = words
        self.word_index = {e.word: e for e in self.words}

    @classmethod
    def load(cls, path):
        if str(path).endswith(".gz"):
            with gzip.GzipFile(path, "r") as f:
                words = [Word.from_dict(e) for e in json.load(f)]
        else:
            with open(path, "r") as f:
                words = [Word.from_dict(e) for e in json.load(f)]
        return cls(words)
    
    @classmethod
    def load_encrypted(cls, path, fernet_key):
        fernet = Fernet(key=fernet_key)

        def decrypt_from_streamy(f):
            byt = fernet.decrypt(f.read())
            words = [Word.from_dict(e) for e in json.loads(byt.decode("utf-8"))]
            return cls(words)

        if str(path).endswith(".gz"):
            with gzip.GzipFile(path, "rb") as f:
                return decrypt_from_streamy(f)
        else:
            with open(path, "rb") as f:
                return decrypt_from_streamy(f)

    def dump(self, path):
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.words], f)

    def dump_encrypted(self, path, fernet_key):
        def encrypt_to_streamy(f):
            fernet = Fernet(key=fernet_key)
            j = json.dumps([e.to_dict() for e in self.words])
            jb = j.encode('utf-8')
            byt = fernet.encrypt(jb)
            f.write(byt)
        
        if str(path).endswith(".gz"):
            with gzip.GzipFile(path, "wb") as f:
                encrypt_to_streamy(f)
        else:
            with open(path, "wb") as f:
                encrypt_to_streamy(f)

    def random(self):
        return random.choice(self.words)

    def by_name(self, name):
        return self.word_index[name]
