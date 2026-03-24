from typing import Literal, Annotated, Union
from pydantic import BaseModel, Field

from .schemas import (
    StepAction, 
    MoveSpotArg, 
    RotateSpotArg,
    FinishTaskArg,
)



# ---------- SCHEMA 0 ----------
# Baseline: {"tool_name": "...", "arguments": {...}]}

SCHEMA_0_SAMPLE = """    
{
    "tool_name": "move_spot",
    "arguments": {
        "meters": 2.0
    }
}
"""

Schema0 = StepAction


# ---------- SCHEMA 1 ----------
# Adds wrapper object: {"action": {...}}

SCHEMA_1_SAMPLE = """    
{
    "action": {
        "tool_name": "rotate_spot",
        "arguments": {
            "degrees": 90
        }
    }
}
"""

class StepSchema1(BaseModel):
    action: StepAction


# ---------- SCHEMA 2 ----------
# Different field names: {"call": {"name": "...", "args": {...}}}

SCHEMA_2_SAMPLE = """    
{
    "call": {
        "name": "move_spot",
        "args": {
            "meters": 2.0
        }
    }
}
"""

class StepSchema2MoveCall(BaseModel):
    name: Literal["move_spot"]
    args: MoveSpotArg

class StepSchema2RotateCall(BaseModel):
    name: Literal["rotate_spot"]
    args: RotateSpotArg

class StepSchema2FinishCall(BaseModel):
    name: Literal["finish_task"]
    args: FinishTaskArg

StepSchema2Call = Annotated[
    Union[
        StepSchema2MoveCall, 
        StepSchema2RotateCall,
        StepSchema2FinishCall
    ],
    Field(discriminator="name"),
]

class StepSchema2(BaseModel):
    call: StepSchema2Call


# ---------- SCHEMA 3 ----------
# Adds wrapper object: {"next_action": {...}}

SCHEMA_3_SAMPLE = """    
{
    "next_action": {
        "tool_name": "move_spot",
        "arguments": {
            "meters": 1.0
        }
    }
}
"""

class StepSchema3(BaseModel):
    next_action: StepAction


#---------- Registry ----------
STEP_SCHEMAS = [
    {
        "id": "st0",
        "schema": Schema0,
        "sample": SCHEMA_0_SAMPLE,
    },
    {
        "id": "st1",
        "schema": StepSchema1,
        "sample": SCHEMA_1_SAMPLE,
    },
    {
        "id": "st2",
        "schema": StepSchema2,
        "sample": SCHEMA_2_SAMPLE,
    },
    {
        "id": "st3",
        "schema": StepSchema3,
        "sample": SCHEMA_3_SAMPLE,
    }
]

STEP_SCHEMA_BY_ID = {
    schema_config["id"]: schema_config
    for schema_config in STEP_SCHEMAS
}