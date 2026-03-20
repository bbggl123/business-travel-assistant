from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from business_travel_assistant.models.trip_state import TripState


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: TripState, **kwargs) -> TripState:
        pass
    
    def _init_cot_reasoning(self) -> List[str]:
        return []
    
    def _add_reasoning_step(self, reasoning: List[str], step: str):
        reasoning.append(f"[{self.name}] {step}")
