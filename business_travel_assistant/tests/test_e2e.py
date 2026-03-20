import pytest
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_travel_assistant.models.trip_state import TripState, IntentType, WorkflowStatus
from business_travel_assistant.workflow.trip_workflow import TripWorkflow
from business_travel_assistant.agents.trip_planning_agent import TripPlanningAgent
from business_travel_assistant.agents.budget_estimation_agent import BudgetEstimationAgent
from business_travel_assistant.agents.approval_generation_agent import ApprovalGenerationAgent, ApprovalForm


class TestTripPlanningAgent:
    def setup_method(self):
        self.agent = TripPlanningAgent()
    
    @pytest.mark.asyncio
    async def test_plan_transport(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        state.travel_mode = "高铁"
        
        result = await self.agent.execute(state)
        
        assert len(result.transport_options) > 0
        for option in result.transport_options:
            assert option.price > 0
            assert option.is_compliant in [True, False]
    
    @pytest.mark.asyncio
    async def test_plan_hotel(self):
        state = TripState(user_id="test")
        state.destination = "上海"
        state.user_level = "初级"
        state.hotel_type = "舒适型"
        
        result = await self.agent.execute(state)
        
        assert len(result.hotel_options) > 0
        for option in result.hotel_options:
            assert option.price_per_night > 0
            assert option.name != ""
    
    @pytest.mark.asyncio
    async def test_plan_with_banquet(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        state.user_level = "初级"
        state.travel_mode = "飞机"
        state.hotel_type = "舒适型"
        state.has_banquet = True
        state.banquet_budget = 150
        
        result = await self.agent.execute(state)
        
        assert len(result.restaurant_options) > 0


class TestBudgetEstimationAgent:
    def setup_method(self):
        self.agent = BudgetEstimationAgent()
    
    @pytest.mark.asyncio
    async def test_estimate_within_budget(self):
        from business_travel_assistant.models.trip_state import TransportOption, HotelOption
        
        state = TripState(user_id="test")
        state.user_level = "初级"
        state.selected_transport = TransportOption(
            type="高铁", provider="中国铁路", flight_no="G1234",
            price=500, is_compliant=True, exceeds_amount=0
        )
        state.selected_hotel = HotelOption(
            name="如家", type="经济型", address="上海",
            price_per_night=180, total_price=180,
            is_compliant=True, exceeds_amount=0
        )
        
        result = await self.agent.execute(state)
        
        assert result.agent_reasoning is not None
    
    @pytest.mark.asyncio
    async def test_estimate_with_exceeds(self):
        from business_travel_assistant.models.trip_state import TransportOption, HotelOption
        
        state = TripState(user_id="test")
        state.user_level = "初级"
        state.selected_transport = TransportOption(
            type="飞机", provider="国航", flight_no="CA1234",
            price=1200, is_compliant=False, exceeds_amount=200
        )
        state.selected_hotel = HotelOption(
            name="豪华酒店", type="高档/豪华型", address="上海",
            price_per_night=800, total_price=800,
            is_compliant=False, exceeds_amount=300
        )
        
        result = await self.agent.execute(state)
        
        assert result.agent_reasoning is not None


class TestApprovalGenerationAgent:
    def setup_method(self):
        self.agent = ApprovalGenerationAgent()
    
    @pytest.mark.asyncio
    async def test_generate_approval_form(self):
        from business_travel_assistant.models.trip_state import TransportOption, HotelOption
        
        state = TripState(user_id="张三")
        state.user_level = "初级/基层员工"
        state.origin = "北京"
        state.destination = "上海"
        state.departure_time = datetime(2026, 3, 21, 8, 0)
        state.return_time = datetime(2026, 3, 23, 17, 0)
        state.trip_purpose = "客户拜访"
        state.selected_transport = TransportOption(
            type="高铁", provider="中国铁路", flight_no="G1234",
            price=553, is_compliant=True, exceeds_amount=0
        )
        state.selected_hotel = HotelOption(
            name="如家快捷酒店", type="经济型", address="上海黄浦区",
            price_per_night=180, total_price=360,
            is_compliant=True, exceeds_amount=0
        )
        
        result = await self.agent.execute(state)
        
        assert result.agent_reasoning is not None
        assert "Approval-Generation-Agent" in result.agent_reasoning
    
    def test_export_form_text(self):
        form = ApprovalForm(
            form_id="TRIP-20260321000001",
            applicant_name="张三",
            applicant_level="初级",
            trip_purpose="客户拜访",
            origin="北京",
            destination="上海",
            subtotal=1000,
            total_exceeds=0,
            items_exceeding_standard=[]
        )
        
        text = self.agent.export_form_text(form)
        
        assert "TRIP-20260321000001" in text
        assert "张三" in text
        assert "北京" in text
        assert "上海" in text
        assert "1000" in text


class TestTripWorkflow:
    def setup_method(self):
        self.workflow = TripWorkflow()
    
    @pytest.mark.asyncio
    async def test_initial_intent_classification(self):
        state = await self.workflow.run("我要从北京去上海出差", "user1")
        
        assert state.user_id == "user1"
        assert state.workflow_status != WorkflowStatus.INITIAL
    
    @pytest.mark.asyncio
    async def test_workflow_generates_questions(self):
        state = await self.workflow.run("我要去上海", "user1")
        
        questions = self.workflow.get_next_questions(state)
        
        assert isinstance(questions, list)
    
    @pytest.mark.asyncio
    async def test_multiple_iterations(self):
        state = TripState(user_id="test")
        state.origin = "北京"
        state.destination = "上海"
        
        state = await self.workflow.process(state, "用户提供了更多信息")
        
        assert state is not None
    
    def test_should_continue(self):
        state = TripState()
        state.workflow_status = WorkflowStatus.INITIAL
        
        assert self.workflow.should_continue(state) == True
        
        state.workflow_status = WorkflowStatus.END
        
        assert self.workflow.should_continue(state) == False
    
    def test_get_next_questions_empty(self):
        state = TripState()
        state.pending_questions = []
        
        questions = self.workflow.get_next_questions(state)
        
        assert questions == []


class TestEndToEndScenarios:
    @pytest.mark.asyncio
    async def test_complete_trip_planning_flow(self):
        workflow = TripWorkflow()
        
        state = await workflow.run("我明天要从北京去上海出差", "user1")
        
        assert state.messages is not None
        assert len(state.messages) > 0
    
    @pytest.mark.asyncio
    async def test_partial_info_collection(self):
        workflow = TripWorkflow()
        
        state = await workflow.run("北京到上海，明天走", "user1")
        
        if state.pending_questions:
            assert len(state.pending_questions) > 0
            assert len(state.pending_questions) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
