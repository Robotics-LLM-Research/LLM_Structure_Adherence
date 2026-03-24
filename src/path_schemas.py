from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field, RootModel

from .schemas import PathAction, MoveSpotArg, RotateSpotArg



# ---------- SCHEMA 0 ----------
# Baseline: {"actions": [...]}

SCHEMA_0_SAMPLE = """    
    {
        "actions": [
            {
                "tool_name": "rotate_spot",
                "arguments": {
                    "degrees": 90
                }
            },
            {
                "tool_name": "move_spot",
                "arguments": {
                    "meters": 2.0
                }
            }
        ]
    }
"""

class PathSchema0(BaseModel):
    actions: list[PathAction]


# ---------- SCHEMA 1 ----------
# Removes wrapper object: [...]

SCHEMA_1_SAMPLE = """    
    [
        {
            "tool_name": "rotate_spot",
            "arguments": {
                "degrees": 90
            }
        },
        {
            "tool_name": "move_spot",
            "arguments": {
                "meters": 2.0
            }
        }
    ]
"""

class PathSchema1(RootModel[list[PathAction]]):
    pass


# ---------- SCHEMA 2 ----------
# Adds required "step" field

SCHEMA_2_SAMPLE = """    
    {
        "actions": [
            {
                "step": 1,
                "tool_name": "rotate_spot",
                "arguments": {
                    "degrees": 90
                }
            },
            {
                "step": 2,
                "tool_name": "move_spot",
                "arguments": {
                    "meters": 2.0
                }
            }
        ]
    }
"""

class PathSchema2MoveStep(BaseModel):
    step: int
    tool_name: Literal["move_spot"]
    arguments: MoveSpotArg

class PathSchema2RotateStep(BaseModel):
    step: int
    tool_name: Literal["rotate_spot"]
    arguments: RotateSpotArg

PathSchema2Action = Annotated[
    Union[PathSchema2MoveStep, PathSchema2RotateStep],
    Field(discriminator="tool_name"),
]

class PathSchema2(BaseModel):
    actions: list[PathSchema2Action]


# ---------- SCHEMA 3 ----------
# Different field names: {"plan": [{"action": ..., "params": ...}]}

SCHEMA_3_SAMPLE = """    
    {
        "plan": [
            {
                "action": "rotate_spot",
                "params": {
                    "degrees": 90
                }
            },
            {
                "action": "move_spot",
                "params": {
                    "meters": 2.0
                }
            }
        ]
    }
"""

class PathSchema3MoveStep(BaseModel):
    action: Literal["move_spot"]
    params: MoveSpotArg

class PathSchema3RotateStep(BaseModel):
    action: Literal["rotate_spot"]
    params: RotateSpotArg 

PathSchema3Action = Annotated[
    Union[PathSchema3MoveStep, PathSchema3RotateStep],
    Field(discriminator="action"),
]

class PathSchema3(BaseModel):
    plan: list[PathSchema3Action]


# ---------- SCHEMA 4 ----------
# Extra nesting: {"steps": [{"call": {...}}]}

SCHEMA_4_SAMPLE = """    
    {
        "steps": [
            {
                "call": {
                    "tool_name": "rotate_spot",
                    "arguments": {
                        "degrees": 90
                    }
                }

            },
            {
                "call": {
                    "tool_name": "move_spot",
                    "arguments": {
                        "meters": 2.0
                    }
                }
            }
        ]
    }
"""

class PathSchema4Step(BaseModel):
    call: PathAction

class PathSchema4(BaseModel):
    steps: list[PathSchema4Step]


# ---------- Registry ----------

PATH_SCHEMAS = [
    {
        "id": "s0",
        "schema": PathSchema0,
        "sample": SCHEMA_0_SAMPLE,
    },
    {
        "id": "s1",
        "schema": PathSchema1,
        "sample": SCHEMA_1_SAMPLE,
    },
    {
        "id": "s2",
        "schema": PathSchema2,
        "sample": SCHEMA_2_SAMPLE,
    },
    {
        "id": "s3",
        "schema": PathSchema3,
        "sample": SCHEMA_3_SAMPLE,
    },
    {
        "id": "s4",
        "schema": PathSchema4,
        "sample": SCHEMA_4_SAMPLE,
    },
]


PATH_SCHEMA_BY_ID = {
    schema_config["id"]: schema_config 
    for schema_config in PATH_SCHEMAS
}