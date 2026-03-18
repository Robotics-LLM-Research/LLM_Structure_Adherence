from typing import Literal, Annotated, Union

from pydantic import BaseModel, Field



# --- Low Level ---
# Move
class MoveSpotArg(BaseModel):
    meters: float

class MoveSpotAction(BaseModel):
    tool_name: Literal["move_spot"]
    arguments: MoveSpotArg

# Rotate
class RotateSpotArg(BaseModel):
    degrees: float

class RotateSpotAction(BaseModel):
    tool_name: Literal["rotate_spot"]
    arguments: RotateSpotArg

# --- High Level ---
Action = Annotated[
    Union[MoveSpotAction, RotateSpotAction],
    Field(discriminator="tool_name")
]

class ActionPlan(BaseModel):
    actions: list[Action]