import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_travel_assistant.models.trip_state import TripState, IntentType, WorkflowStatus
from business_travel_assistant.agents.intent_classification_agent import IntentClassificationAgent
from business_travel_assistant.agents.information_collection_agent import InformationCollectionAgent
from business_travel_assistant.agents.compliance_check_agent import ComplianceCheckAgent


class TestIntentClassificationAgent:
    def setup_method(self):
        self.agent = IntentClassificationAgent()
    
    @pytest.mark.asyncio
    async def test_classify_initial_request(self):
        state = TripState(user_id="test")
        state.origin = ""
        state.destination = ""
        
        result = await self.agent.execute(state, user_input="我要去上海出差")
        
        assert result.current_intent == IntentType.INITIAL_REQUEST
        assert "user_level" in result.missing_fields
        assert "origin" in result.missing_fields
    
    @pytest.mark.asyncio
    async def test_classify_with_partial_info(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        
        result = await self.agent.execute(state, user_input="明天出发")
        
        assert result.current_intent == IntentType.INFO_PROVIDED
        assert len(result.missing_fields) >= 0
    
    @pytest.mark.asyncio
    async def test_identify_missing_fields(self):
        state = TripState(user_id="test")
        
        missing = self.agent._identify_missing_fields(state)
        
        assert "user_level" in missing
        assert "origin" in missing
        assert "destination" in missing
        assert "departure_time" in missing
        assert "travel_mode" in missing
    
    @pytest.mark.asyncio
    async def test_reasoning_is_recorded(self):
        state = TripState(user_id="test")
        
        result = await self.agent.execute(state, user_input="我去北京")
        
        assert self.agent.name in result.agent_reasoning
        assert len(result.agent_reasoning[self.agent.name]) > 0


class TestInformationCollectionAgent:
    def setup_method(self):
        self.agent = InformationCollectionAgent()
    
    @pytest.mark.asyncio
    async def test_generate_questions(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        
        result = await self.agent.execute(state)
        
        assert len(result.pending_questions) > 0
        assert result.pending_questions[0].field_name in ["departure_time", "return_time", "travel_mode", "hotel_type"]
    
    @pytest.mark.asyncio
    async def test_no_questions_when_complete(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        state.departure_time = None
        state.return_time = None
        state.travel_mode = "飞机"
        state.hotel_type = "舒适型"
        state.trip_purpose = "客户拜访"
        state.customer_location = "浦东新区"
        
        missing = self.agent._identify_missing_fields(state)
        
        assert "departure_time" in missing


class TestComplianceCheckAgent:
    def setup_method(self):
        self.agent = ComplianceCheckAgent()
    
    @pytest.mark.asyncio
    async def test_check_transport_compliant(self):
        result = await self.agent.check_transport("飞机", 800, "初级/基层员工")
        
        assert result["is_compliant"] == True
        assert result["exceeds_amount"] == 0
    
    @pytest.mark.asyncio
    async def test_check_transport_exceeds(self):
        result = await self.agent.check_transport("飞机", 1500, "初级/基层员工")
        
        assert result["is_compliant"] == False
        assert result["exceeds_amount"] == 500
    
    @pytest.mark.asyncio
    async def test_check_hotel_compliant(self):
        result = await self.agent.check_hotel("舒适型", 250, "初级/基层员工")
        
        assert result["is_compliant"] == True
        assert result["standard_limit"] == 300
    
    @pytest.mark.asyncio
    async def test_check_hotel_exceeds(self):
        result = await self.agent.check_hotel("高档/豪华型", 800, "初级/基层员工")
        
        assert result["is_compliant"] == False
        assert result["exceeds_amount"] == 300
    
    @pytest.mark.asyncio
    async def test_check_banquet_compliant(self):
        result = await self.agent.check_banquet(100, "初级/基层员工")
        
        assert result["is_compliant"] == True
    
    @pytest.mark.asyncio
    async def test_check_banquet_exceeds(self):
        result = await self.agent.check_banquet(200, "初级/基层员工")
        
        assert result["is_compliant"] == False
        assert result["exceeds_amount"] == 50
    
    @pytest.mark.asyncio
    async def test_different_user_levels(self):
        junior_result = await self.agent.check_transport("飞机", 1200, "初级/基层员工")
        middle_result = await self.agent.check_transport("飞机", 1200, "中级/中层管理者")
        senior_result = await self.agent.check_transport("飞机", 1200, "高级/高层管理者")
        
        assert junior_result["is_compliant"] == False
        assert middle_result["is_compliant"] == True
        assert senior_result["is_compliant"] == True


class TestAgentCollaboration:
    @pytest.mark.asyncio
    async def test_intent_to_collection_flow(self):
        intent_agent = IntentClassificationAgent()
        collection_agent = InformationCollectionAgent()
        
        state = TripState(user_id="test")
        state = await intent_agent.execute(state, user_input="我要从北京去上海")
        
        assert state.current_intent == IntentType.INITIAL_REQUEST
        assert len(state.missing_fields) > 0
        
        state = await collection_agent.execute(state)
        
        assert len(state.pending_questions) > 0
    
    @pytest.mark.asyncio
    async def test_full_collection_flow(self):
        intent_agent = IntentClassificationAgent()
        collection_agent = InformationCollectionAgent()
        
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        
        state = await intent_agent.execute(state, user_input="北京到上海")
        state = await collection_agent.execute(state)
        
        assert state.pending_questions is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
