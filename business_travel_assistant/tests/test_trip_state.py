import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from business_travel_assistant.models.trip_state import TripState, WorkflowStatus, IntentType, Question


class TestTripState:
    def test_trip_state_initialization(self):
        state = TripState(user_id="test_user")
        
        assert state.user_id == "test_user"
        assert state.workflow_status == WorkflowStatus.INITIAL
        assert state.current_intent == IntentType.UNKNOWN
        assert state.messages == []
    
    def test_add_reasoning(self):
        state = TripState()
        
        state.add_reasoning("TestAgent", "Step 1: Test reasoning")
        
        assert "TestAgent" in state.agent_reasoning
        assert len(state.agent_reasoning["TestAgent"]) == 1
        assert "Step 1: Test reasoning" in state.agent_reasoning["TestAgent"]
    
    def test_add_message(self):
        state = TripState()
        
        state.add_message("user", "Hello")
        
        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == "Hello"
        assert "timestamp" in state.messages[0]
    
    def test_get_all_fields(self):
        state = TripState(
            user_id="test",
            user_level="初级",
            origin="北京",
            destination="上海"
        )
        
        fields = state.get_all_fields()
        
        assert fields["user_id"] == "test"
        assert fields["user_level"] == "初级"
        assert fields["origin"] == "北京"
        assert fields["destination"] == "上海"
    
    def test_workflow_status_enum(self):
        assert WorkflowStatus.INITIAL.value == "INITIAL"
        assert WorkflowStatus.INTENT_CLASSIFICATION.value == "INTENT_CLASSIFICATION"
        assert WorkflowStatus.COLLECTING_INFO.value == "COLLECTING_INFO"
        assert WorkflowStatus.PLANNING.value == "PLANNING"
        assert WorkflowStatus.COMPLETED.value == "COMPLETED"
    
    def test_intent_type_enum(self):
        assert IntentType.INITIAL_REQUEST.value == "INITIAL_REQUEST"
        assert IntentType.INFO_PROVIDED.value == "INFO_PROVIDED"
        assert IntentType.CONFIRM.value == "CONFIRM"
        assert IntentType.UNKNOWN.value == "UNKNOWN"


class TestWorkflowTransitions:
    def test_initial_to_intent_classification(self):
        state = TripState()
        
        assert state.workflow_status == WorkflowStatus.INITIAL
        
        state.workflow_status = WorkflowStatus.INTENT_CLASSIFICATION
        
        assert state.workflow_status == WorkflowStatus.INTENT_CLASSIFICATION
    
    def test_intent_classification_to_collecting_info(self):
        state = TripState()
        state.workflow_status = WorkflowStatus.INTENT_CLASSIFICATION
        state.missing_fields = ["user_level", "origin"]
        
        state.workflow_status = WorkflowStatus.COLLECTING_INFO
        
        assert state.workflow_status == WorkflowStatus.COLLECTING_INFO
        assert len(state.missing_fields) == 2
    
    def test_collecting_info_to_planning(self):
        state = TripState()
        state.workflow_status = WorkflowStatus.COLLECTING_INFO
        state.missing_fields = []
        state.pending_questions = []
        
        state.workflow_status = WorkflowStatus.PLANNING
        
        assert state.workflow_status == WorkflowStatus.PLANNING
        assert len(state.missing_fields) == 0
    
    def test_planning_to_completed(self):
        state = TripState()
        state.workflow_status = WorkflowStatus.PLANNING
        state.selected_transport = None
        state.selected_hotel = None
        
        state.workflow_status = WorkflowStatus.APPROVAL_GENERATION
        
        assert state.workflow_status == WorkflowStatus.APPROVAL_GENERATION
        
        state.workflow_status = WorkflowStatus.COMPLETED
        
        assert state.workflow_status == WorkflowStatus.COMPLETED
    
    def test_full_workflow_path(self):
        state = TripState()
        
        assert state.workflow_status == WorkflowStatus.INITIAL
        
        state.workflow_status = WorkflowStatus.INTENT_CLASSIFICATION
        state.current_intent = IntentType.INITIAL_REQUEST
        state.missing_fields = ["user_level"]
        
        assert state.workflow_status == WorkflowStatus.INTENT_CLASSIFICATION
        assert state.current_intent == IntentType.INITIAL_REQUEST


class TestQuestionModel:
    def test_question_creation(self):
        question = Question(
            field_name="user_level",
            question_text="请问您的职级是？",
            options=["初级", "中级", "高级"],
            allow_custom=True
        )
        
        assert question.field_name == "user_level"
        assert question.question_text == "请问您的职级是？"
        assert len(question.options) == 3
        assert question.allow_custom == True
        assert question.reasoning == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
