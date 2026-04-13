from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

from .base import StepAction


# ---------- Observation Keys ----------
ObservationKey = Literal[
    "wall_ahead",
    "left_clear",
    "right_clear",
    "at_goal",
]


# ---------- Leaf Nodes ----------
class BTConditionNode(BaseModel):
    type: Literal["condition"]
    observation: ObservationKey
    expected: bool = True

class BTActionNode(BaseModel):
    type: Literal["action"]
    call: StepAction


# ---------- Control Nodes ----------
class BTSequenceNode(BaseModel):
    type: Literal["sequence"]
    children: list["BTNode"] = Field(min_length=1)

class BTFallbackNode(BaseModel):
    type: Literal["fallback"]
    children: list["BTNode"] = Field(min_length=1)


# ---------- Node Unions ----------
BTNode = Annotated[
    Union[
        BTConditionNode,
        BTActionNode,
        BTSequenceNode,
        BTFallbackNode,
    ],
    Field(discriminator="type"),
]

BTControlNode = Annotated[
    Union[
        BTSequenceNode,
        BTFallbackNode,
    ],
    Field(discriminator="type"),
]


# ---------- Schema ----------
WALL_BT_SCHEMA_SAMPLE = """
{
    "schema_version": "bt_json_v1",
    "root": {
        "type": "fallback",
        "children": [
            {
                "type": "sequence",
                "children": [
                    {
                        "type": "condition",
                        "observation": "at_goal",
                        "expected": true
                    },
                    {
                        "type": "action",
                        "call": {
                            "tool_name": "finish_task",
                            "arguments": {}
                        }
                    }
                ]
            },
            {
                "type": "action",
                "call": {
                    "tool_name": "call_llm",
                    "arguments": {}
                }
            }
        ]
    }
}
"""

class WallBTSchema(BaseModel):
    schema_version: Literal["bt_json_v1"]
    root: BTControlNode

BTSequenceNode.model_rebuild()
BTFallbackNode.model_rebuild()
WallBTSchema.model_rebuild()


# ---------- Single Config ----------
WALL_BT_SCHEMA_CONFIG = {
    "id": "bt",
    "schema": WallBTSchema,
    "sample": WALL_BT_SCHEMA_SAMPLE,
}