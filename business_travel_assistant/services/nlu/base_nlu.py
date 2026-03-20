from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    entity_type: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int


@dataclass
class SemanticParseResult:
    original_text: str
    intent: str
    entities: List[ExtractedEntity]
    confidence: float
    reasoning_steps: List[str]
    raw_output: Dict[str, Any]


class BaseNLUService(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def parse(self, text: str, context: Optional[Dict[str, Any]] = None) -> SemanticParseResult:
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
        pass
    
    @abstractmethod
    async def classify_intent(self, text: str, candidate_intents: Optional[List[str]] = None) -> Dict[str, float]:
        pass
