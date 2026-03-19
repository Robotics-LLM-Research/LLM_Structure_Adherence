from typing import Literal, Annotated, Union

from pydantic import BaseModel, Field, RootModel



# ---------- Internal Schema ----------
class MoveSpotArg(BaseModel):
    meters: float

class RotateSpotArg(BaseModel):
    degrees: float

class MoveSpotAction(BaseModel):
    tool_name: Literal["move_spot"]
    arguments: MoveSpotArg

class RotateSpotAction(BaseModel):
    tool_name: Literal["rotate_spot"]
    arguments: RotateSpotArg

Action = Annotated[
    Union[MoveSpotAction, RotateSpotAction],
    Field(discriminator="tool_name")
]

class ActionPlan(BaseModel):
    actions: list[Action]


# ---------- SCHEMA 1 ----------
# Baseline: {"actions": [...]}

SCHEMA_1_SAMPLE = """    
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

class Schema1(BaseModel):
    actions: list[Action]


# ---------- SCHEMA 2 ----------
# Removes wrapper object: [...]

SCHEMA_2_SAMPLE = """    
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

class Schema2(RootModel[list[Action]]):
    pass


# ---------- SCHEMA 3 ----------
# Adds required "step" field

SCHEMA_3_SAMPLE = """    
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

class Schema3MoveStep(BaseModel):
    step: int
    tool_name: Literal["move_spot"]
    arguments: MoveSpotArg

class Schema3RotateStep(BaseModel):
    step: int
    tool_name: Literal["rotate_spot"]
    arguments: RotateSpotArg

Schema3Action = Annotated[
    Union[Schema3MoveStep, Schema3RotateStep],
    Field(discriminator="tool_name"),
]

class Schema3(BaseModel):
    actions: list[Schema3Action]


# ---------- SCHEMA 4 ----------
# Different field names: {"plan": [{"action": ..., "params": ...}]}

SCHEMA_4_SAMPLE = """    
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

class Schema4MoveStep(BaseModel):
    action: Literal["move_spot"]
    params: MoveSpotArg

class Schema4RotateStep(BaseModel):
    action: Literal["rotate_spot"]
    params: RotateSpotArg 

Schema4Action = Annotated[
    Union[Schema4MoveStep, Schema4RotateStep],
    Field(discriminator="action"),
]

class Schema4(BaseModel):
    plan: list[Schema4Action]


# ---------- SCHEMA 5 ----------
# Extra nesting: {"steps": [{"call": {...}}]}

SCHEMA_5_SAMPLE = """    
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

class Schema5Step(BaseModel):
    call: Action

class Schema5(BaseModel):
    steps: list[Schema5Step]


# ---------- Registry ----------

SCHEMAS = [
    {
        "id": "s0",
        "schema": Schema1,
        "sample": SCHEMA_1_SAMPLE,
    },
    {
        "id": "s1",
        "schema": Schema2,
        "sample": SCHEMA_2_SAMPLE,
    },
    {
        "id": "s2",
        "schema": Schema3,
        "sample": SCHEMA_3_SAMPLE,
    },
    {
        "id": "s3",
        "schema": Schema4,
        "sample": SCHEMA_4_SAMPLE,
    },
    {
        "id": "s4",
        "schema": Schema5,
        "sample": SCHEMA_5_SAMPLE,
    },
]


SCHEMA_BY_ID = {
    schema_config["id"]: schema_config 
    for schema_config in SCHEMAS
}