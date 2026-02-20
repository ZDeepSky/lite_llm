from dataclasses import dataclass,asdict


@dataclass
class BaseConfig:
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


