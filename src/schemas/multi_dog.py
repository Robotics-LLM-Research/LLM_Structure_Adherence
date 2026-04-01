from typing import Union, Literal, Annotated
from pydantic import Field, BaseModel

from .base import MoveSpotArg, RotateSpotArg


# ----- Per-dog Action Schemas -----
class MultiDogMoveAction(BaseModel):
    tool_name: Literal["move_spot"]
    args: MoveSpotArg


class MultiDogRotateAction(BaseModel):
    tool_name: Literal["rotate_spot"]
    args: RotateSpotArg


MultiDogDogAction = Annotated[
    Union[MultiDogMoveAction, MultiDogRotateAction],
    Field(discriminator="tool_name"),
]


# ----- Schema -----
MULTI_DOG_STEP_SCHEMA_SAMPLE = """
{
    "dog1": {
        "tool_name": "move_spot",
        "args": {
            "meters": 0.4
        }
    },
    "dog2": {
        "tool_name": "rotate_spot",
        "args": {
            "degrees": -20
        }
    },
    "dog3": {
        "tool_name": "move_spot",
        "args": {
            "meters": 0.3
        }
    },
    "dog4": {
        "tool_name": "rotate_spot",
        "args": {
            "degrees": 30
        }
    },
    "dog5": {
        "tool_name": "move_spot",
        "args": {
            "meters": 0.2
        }
    }
}
"""


class MultiDogStepSchema(BaseModel):
    dog1: MultiDogDogAction
    dog2: MultiDogDogAction
    dog3: MultiDogDogAction
    dog4: MultiDogDogAction
    dog5: MultiDogDogAction


# ----- Single Config -----
MULTI_DOG_STEP_SCHEMA_CONFIG = {
    "id": "md_st",
    "schema": MultiDogStepSchema,
    "sample": MULTI_DOG_STEP_SCHEMA_SAMPLE,
}
