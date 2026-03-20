import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_travel_assistant.services.nlu.base_nlu import BaseNLUService, SemanticParseResult, ExtractedEntity
from business_travel_assistant.services.nlu.mock_nlu_service import MockNLUService
from business_travel_assistant.services.nlu.nlu_manager import NLUServiceManager, nlu_manager


class TestBaseNLUService:
    def test_nlu_service_interface定义(self):
        assert hasattr(BaseNLUService, 'parse')
        assert hasattr(BaseNLUService, 'extract_entities')
        assert hasattr(BaseNLUService, 'classify_intent')


class TestMockNLUService:
    def setup_method(self):
        self.nlu = MockNLUService()
    
    @pytest.mark.asyncio
    async def test_parse_simple_trip_request(self):
        result = await self.nlu.parse("我要从北京去上海出差")
        
        assert isinstance(result, SemanticParseResult)
        assert result.original_text == "我要从北京去上海出差"
        assert result.intent in ["TRIP_REQUEST", "INFO_PROVIDED", "UNKNOWN"]
        assert len(result.entities) > 0
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_parse_with_multiple_locations(self):
        result = await self.nlu.parse("我从北京去上海，明天出发")
        
        locations = [e for e in result.entities if e.entity_type == "LOCATION"]
        assert len(locations) >= 2
        location_values = [e.value for e in locations]
        assert "北京" in location_values
        assert "上海" in location_values
    
    @pytest.mark.asyncio
    async def test_parse_travel_mode(self):
        result = await self.nlu.parse("我要坐飞机去上海")
        
        travel_modes = [e for e in result.entities if e.entity_type == "TRAVEL_MODE"]
        assert len(travel_modes) > 0
        assert travel_modes[0].value == "飞机"
    
    @pytest.mark.asyncio
    async def test_parse_hotel_type(self):
        result = await self.nlu.parse("我需要住高档酒店")
        
        hotel_types = [e for e in result.entities if e.entity_type == "HOTEL_TYPE"]
        assert len(hotel_types) > 0
    
    @pytest.mark.asyncio
    async def test_parse_user_level(self):
        result = await self.nlu.parse("我是基层员工，要去出差")
        
        user_levels = [e for e in result.entities if e.entity_type == "USER_LEVEL"]
        assert len(user_levels) > 0
    
    @pytest.mark.asyncio
    async def test_parse_time_relative(self):
        result = await self.nlu.parse("明天我要出差")
        
        times = [e for e in result.entities if e.entity_type == "TIME_RELATIVE"]
        assert len(times) > 0
    
    @pytest.mark.asyncio
    async def test_extract_entities(self):
        text = "我从北京去上海，明天出发，想住高档酒店"
        entities = await self.nlu.extract_entities(text)
        
        assert len(entities) > 0
        entity_types = set(e.entity_type for e in entities)
        assert "LOCATION" in entity_types
    
    @pytest.mark.asyncio
    async def test_classify_intent_trip_request(self):
        result = await self.nlu.classify_intent("我要从北京去上海出差")
        
        assert isinstance(result, dict)
        assert "TRIP_REQUEST" in result
        assert result["TRIP_REQUEST"] > 0
    
    @pytest.mark.asyncio
    async def test_classify_intent_confirm(self):
        result = await self.nlu.classify_intent("好的，确认")
        
        assert result["CONFIRM"] > 0 or result["UNKNOWN"] > 0
    
    @pytest.mark.asyncio
    async def test_entity_confidence_scores(self):
        entities = await self.nlu.extract_entities("北京")
        
        for entity in entities:
            assert 0 <= entity.confidence <= 1


class TestNLUServiceManager:
    def setup_method(self):
        self.manager = NLUServiceManager()
    
    def test_singleton_pattern(self):
        manager1 = NLUServiceManager()
        manager2 = NLUServiceManager()
        assert manager1 is manager2
    
    def test_initialize_with_service(self):
        custom_service = MockNLUService()
        self.manager.initialize(custom_service)
        assert self.manager._nlu_service is custom_service
    
    def test_initialize_without_service(self):
        self.manager.initialize()
        assert self.manager._nlu_service is not None
        assert isinstance(self.manager._nlu_service, MockNLUService)
    
    @pytest.mark.asyncio
    async def test_parse_via_manager(self):
        self.manager.initialize()
        result = await self.manager.parse("我要去上海出差")
        
        assert isinstance(result, SemanticParseResult)
    
    @pytest.mark.asyncio
    async def test_extract_trip_info(self):
        self.manager.initialize()
        info = await self.manager.extract_trip_info("我从北京去上海出差")
        
        assert "intent" in info
        assert "confidence" in info
        assert "entities" in info
        assert "structured_data" in info
        assert "reasoning" in info


class TestEntityExtraction:
    def setup_method(self):
        self.nlu = MockNLUService()
    
    @pytest.mark.asyncio
    async def test_extract_location_entities(self):
        entities = await self.nlu.extract_entities("北京到上海的出差")
        
        locations = [e for e in entities if e.entity_type == "LOCATION"]
        assert len(locations) >= 2
    
    @pytest.mark.asyncio
    async def test_extract_time_entities(self):
        entities = await self.nlu.extract_entities("明天下午出发")
        
        time_entities = [e for e in entities if e.entity_type in ["TIME_RELATIVE", "TIME_OF_DAY"]]
        assert len(time_entities) >= 1
    
    @pytest.mark.asyncio
    async def test_extract_budget(self):
        entities = await self.nlu.extract_entities("预算1000元")
        
        budgets = [e for e in entities if e.entity_type == "BUDGET"]
        assert len(budgets) >= 1
        assert budgets[0].value == "1000"
    
    @pytest.mark.asyncio
    async def test_extract_banquet_info(self):
        entities = await self.nlu.extract_entities("需要宴请客户")
        
        banquets = [e for e in entities if e.entity_type == "BANQUET_REQUIRED"]
        assert len(banquets) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
