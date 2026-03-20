from typing import Dict, Any
from business_travel_assistant.models.trip_state import TripState, WorkflowStatus, IntentType
from business_travel_assistant.agents.intent_classification_agent import IntentClassificationAgent
from business_travel_assistant.agents.information_collection_agent import InformationCollectionAgent
from business_travel_assistant.agents.trip_planning_agent import TripPlanningAgent
from business_travel_assistant.agents.budget_estimation_agent import BudgetEstimationAgent
from business_travel_assistant.agents.approval_generation_agent import ApprovalGenerationAgent


class TripWorkflow:
    def __init__(self):
        self.intent_agent = IntentClassificationAgent()
        self.collection_agent = InformationCollectionAgent()
        self.planning_agent = TripPlanningAgent()
        self.budget_agent = BudgetEstimationAgent()
        self.approval_agent = ApprovalGenerationAgent()
    
    async def process(self, state: TripState, user_input: str = "", **kwargs) -> TripState:
        state.add_message("user", user_input)
        
        if state.workflow_status == WorkflowStatus.INITIAL:
            state = await self._classify_intent(state, user_input)
        
        if state.current_intent == IntentType.INITIAL_REQUEST:
            if state.workflow_status == WorkflowStatus.INITIAL:
                state.workflow_status = WorkflowStatus.INTENT_CLASSIFICATION
            if state.workflow_status == WorkflowStatus.INTENT_CLASSIFICATION:
                state = await self._classify_intent(state, user_input)
            if state.missing_fields:
                state.workflow_status = WorkflowStatus.COLLECTING_INFO
        
        elif state.current_intent == IntentType.INFO_PROVIDED:
            if state.workflow_status == WorkflowStatus.COLLECTING_INFO and state.pending_questions:
                pass
            elif state.missing_fields:
                state.workflow_status = WorkflowStatus.COLLECTING_INFO
            else:
                state.workflow_status = WorkflowStatus.PLANNING
        
        elif state.current_intent == IntentType.CONFIRM:
            state.workflow_status = WorkflowStatus.APPROVAL_GENERATION
        
        if state.workflow_status == WorkflowStatus.COLLECTING_INFO and not state.pending_questions:
            state = await self.collection_agent.execute(state, user_input=user_input)
        
        if state.workflow_status == WorkflowStatus.PLANNING and not state.transport_options:
            state = await self.planning_agent.execute(state)
        
        if state.workflow_status == WorkflowStatus.APPROVAL_GENERATION:
            state = await self.budget_agent.execute(state)
            state = await self.approval_agent.execute(state)
            state.workflow_status = WorkflowStatus.COMPLETED
        
        if state.workflow_status == WorkflowStatus.COMPLETED:
            state.workflow_status = WorkflowStatus.END
        
        return state
    
    async def _classify_intent(self, state: TripState, user_input: str) -> TripState:
        return await self.intent_agent.execute(state, user_input=user_input)
    
    def should_continue(self, state: TripState) -> bool:
        return state.workflow_status != WorkflowStatus.END and state.workflow_status != WorkflowStatus.COMPLETED
    
    def get_next_questions(self, state: TripState) -> list:
        if state.pending_questions:
            return [q.question_text for q in state.pending_questions[:3]]
        return []
    
    async def run(self, user_input: str, user_id: str = "user1") -> TripState:
        state = TripState(user_id=user_id)
        
        max_iterations = 10
        iteration = 0
        
        while self.should_continue(state) and iteration < max_iterations:
            state = await self.process(state, user_input)
            iteration += 1
            
            if state.pending_questions and iteration < max_iterations:
                break
        
        return state
