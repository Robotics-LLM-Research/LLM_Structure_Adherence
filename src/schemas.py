from typing import Union, Literal, Annotated
from pydantic import Field, BaseModel



# ----- Argument Schemas -----
class MoveSpotArg(BaseModel):
    meters: float

class RotateSpotArg(BaseModel):
    degrees: float

class FinishTaskArg(BaseModel):
    pass


# ----- Action Schemas -----
class MoveSpotAction(BaseModel):
    tool_name: Literal["move_spot"]
    arguments: MoveSpotArg

class RotateSpotAction(BaseModel):
    tool_name: Literal["rotate_spot"]
    arguments: RotateSpotArg

class FinishTaskAction(BaseModel):
    tool_name: Literal["finish_task"]
    arguments: FinishTaskArg


# ----- Union Schemas -----
PathAction = Annotated[
    Union[MoveSpotAction, RotateSpotAction],
    Field(discriminator="tool_name"),
]

StepAction = Annotated[
    Union[MoveSpotAction, RotateSpotAction, FinishTaskAction],
    Field(discriminator="tool_name"),
]


# ----- Plan Schema -----
class ActionPlan(BaseModel):
    actions: list[PathAction]