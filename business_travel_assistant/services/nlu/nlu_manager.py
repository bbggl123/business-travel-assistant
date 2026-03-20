from typing import Dict, Optional
from business_travel_assistant.services.nlu.base_nlu import BaseNLUService, SemanticParseResult
from business_travel_assistant.services.nlu.mock_nlu_service import MockNLUService


class NLUServiceManager:
    _instance = None
    _nlu_service: Optional[BaseNLUService] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self, service: Optional[BaseNLUService] = None):
        if service is None:
            service = MockNLUService()
        self._nlu_service = service
    
    @property
    def service(self) -> BaseNLUService:
        if self._nlu_service is None:
            self.initialize()
        return self._nlu_service
    
    async def parse(self, text: str, context: Optional[Dict] = None) -> SemanticParseResult:
        return await self.service.parse(text, context)
    
    async def extract_trip_info(self, text: str) -> Dict:
        result = await self.parse(text)
        
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": [
                {
                    "type": e.entity_type,
                    "value": e.value,
                    "confidence": e.confidence
                }
                for e in result.entities
            ],
            "structured_data": result.raw_output,
            "reasoning": result.reasoning_steps
        }


nlu_manager = NLUServiceManager()
