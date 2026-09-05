from dataclasses import dataclass
from typing import TypedDict, Optional, Literal

from pydantic import BaseModel, Field


class State(BaseModel):
    topic: str
    draft: str | None
    revision_count: int | None
    feedback: str | None

class Prompt(TypedDict):
    system: str | None
    user: str | None


@dataclass(frozen=True)
class UsingDataClass:
    model: str | None
    temperature: int = Field(default=5, min_length=1, description="The temperature in degrees Celsius")
    top_k: Optional[int] = None
    currency: Literal["usd", "eur"] = Field(default="usd", description="The currency to use")



if __name__ == "__main__":
    # state = State(topic="pydantic_exercise", draft="initial", revision_count=1, feedback="feedback")
    # state = State(topic="pydantic_exercise", draft="pydantic_exercise", revision_count=None)
    # prompt = Prompt(system="system")
    using_data = UsingDataClass(model="groq", temperature=5, top_k=5)
    print(using_data)

