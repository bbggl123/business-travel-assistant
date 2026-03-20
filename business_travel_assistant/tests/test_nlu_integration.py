import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_travel_assistant.models.trip_state import TripState, IntentType
from business_travel_assistant.workflow.trip_workflow import TripWorkflow
from business_travel_assistant.agents.enhanced_intent_classification_agent import EnhancedIntentClassificationAgent


class TestEnhancedIntentClassificationAgent:
    def setup_method(self):
        self.agent = EnhancedIntentClassificationAgent()
    
    @pytest.mark.asyncio
    async def test_parse_origin_and_destination(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我从北京去上海出差")
        
        assert result.origin == "北京"
        assert result.destination == "上海"
    
    @pytest.mark.asyncio
    async def test_parse_travel_mode(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我要坐高铁去上海")
        
        assert result.travel_mode == "火车"
    
    @pytest.mark.asyncio
    async def test_parse_hotel_type(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我想住高档酒店")
        
        assert result.hotel_type == "高档/豪华型"
    
    @pytest.mark.asyncio
    async def test_parse_user_level(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我是基层员工")
        
        assert result.user_level == "初级/基层员工"
    
    @pytest.mark.asyncio
    async def test_parse_trip_purpose(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="去拜访客户")
        
        assert result.trip_purpose == "客户拜访"
    
    @pytest.mark.asyncio
    async def test_parse_banquet_requirement(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="需要宴请客户")
        
        assert result.has_banquet == True
    
    @pytest.mark.asyncio
    async def test_parse_complete_trip_request(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我是基层员工，要从北京去上海出差，明天出发，想住高档酒店，需要宴请客户")
        
        assert result.origin == "北京"
        assert result.destination == "上海"
        assert result.user_level == "初级/基层员工"
        assert result.travel_mode == ""
        assert result.has_banquet == True
        assert result.missing_fields is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_is_recorded(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="从北京去上海")
        
        assert "Enhanced-Intent-Classification-Agent" in result.agent_reasoning
        assert len(result.agent_reasoning["Enhanced-Intent-Classification-Agent"]) > 0


class TestEnhancedWorkflow:
    def setup_method(self):
        self.workflow = TripWorkflow(use_enhanced_nlu=True)
    
    @pytest.mark.asyncio
    async def test_workflow_with_enhanced_nlu(self):
        state = await self.workflow.run("我要从北京去上海出差", "user1")
        
        assert state.origin == "北京"
        assert state.destination == "上海"
        assert state.current_intent == IntentType.INITIAL_REQUEST
    
    @pytest.mark.asyncio
    async def test_workflow_extracts_multiple_entities(self):
        state = await self.workflow.run("基层员工从北京去上海，住高档酒店", "user1")
        
        assert state.origin == "北京"
        assert state.destination == "上海"
        assert state.user_level == "初级/基层员工"
        assert state.hotel_type == "高档/豪华型"
    
    @pytest.mark.asyncio
    async def test_workflow_with_banquet(self):
        state = await self.workflow.run("从北京去上海出差，需要宴请客户", "user1")
        
        assert state.has_banquet == True
        assert state.destination == "上海"


class TestNLUIntegration:
    @pytest.mark.asyncio
    async def test_nlu_updates_state_correctly(self):
        agent = EnhancedIntentClassificationAgent()
        state = TripState(user_id="test")
        
        result = await agent.execute(state, user_input="我是中级经理，要从深圳飞往北京，住舒适型酒店")
        
        assert result.user_level == "中级/中层管理者"
        assert result.origin in ["深圳", "北京"]
        assert result.destination in ["深圳", "北京"]
        assert result.origin != result.destination
        assert result.travel_mode == "飞机"
        assert result.hotel_type == "舒适型"
    
    @pytest.mark.asyncio
    async def test_missing_fields_after_nlu_parsing(self):
        agent = EnhancedIntentClassificationAgent()
        state = TripState(user_id="test")
        
        result = await agent.execute(state, user_input="我去上海出差")
        
        assert result.origin == ""
        assert result.destination == "上海"
        assert len(result.missing_fields) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
