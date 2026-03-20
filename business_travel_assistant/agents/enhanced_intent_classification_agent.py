from typing import Dict, Any, List
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, IntentType
from business_travel_assistant.services.nlu.nlu_manager import nlu_manager


class EnhancedIntentClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Enhanced-Intent-Classification-Agent")
        self.nlu = nlu_manager
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        user_input = kwargs.get("user_input", "")
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, f"Step 1: 接收到用户输入: {user_input}")
        
        self._add_reasoning_step(reasoning, "Step 2: 调用NLU服务进行语义解析")
        nlu_result = await self.nlu.parse(user_input)
        
        self._add_reasoning_step(reasoning, f"NLU解析结果:")
        self._add_reasoning_step(reasoning, f"  - 意图: {nlu_result.intent} (置信度: {nlu_result.confidence:.2f})")
        self._add_reasoning_step(reasoning, f"  - 实体数量: {len(nlu_result.entities)}")
        
        self._add_reasoning_step(reasoning, "Step 3: 更新TripState中的信息")
        
        structured_data = nlu_result.raw_output
        
        if structured_data.get("origin"):
            state.origin = structured_data["origin"]
            self._add_reasoning_step(reasoning, f"  - 提取出发地: {state.origin}")
        
        if structured_data.get("destination"):
            state.destination = structured_data["destination"]
            self._add_reasoning_step(reasoning, f"  - 提取目的地: {state.destination}")
        
        if structured_data.get("travel_mode"):
            state.travel_mode = structured_data["travel_mode"]
            self._add_reasoning_step(reasoning, f"  - 提取出行方式: {state.travel_mode}")
        
        if structured_data.get("hotel_type"):
            state.hotel_type = structured_data["hotel_type"]
            self._add_reasoning_step(reasoning, f"  - 提取住宿类型: {state.hotel_type}")
        
        if structured_data.get("user_level"):
            state.user_level = structured_data["user_level"]
            self._add_reasoning_step(reasoning, f"  - 提取用户职级: {state.user_level}")
        
        if structured_data.get("trip_purpose"):
            state.trip_purpose = structured_data["trip_purpose"]
            self._add_reasoning_step(reasoning, f"  - 提取出差目的: {state.trip_purpose}")
        
        if structured_data.get("has_banquet") is not None:
            state.has_banquet = structured_data["has_banquet"]
            self._add_reasoning_step(reasoning, f"  - 提取宴请需求: {state.has_banquet}")
        
        if structured_data.get("banquet_budget"):
            state.banquet_budget = structured_data["banquet_budget"]
            self._add_reasoning_step(reasoning, f"  - 提取宴请预算: {state.banquet_budget}")
        
        self._add_reasoning_step(reasoning, "Step 4: 识别缺失字段")
        missing_fields = self._identify_missing_fields(state)
        self._add_reasoning_step(reasoning, f"  - 缺失字段: {missing_fields}")
        
        self._add_reasoning_step(reasoning, "Step 5: 确定最终意图")
        intent = self._map_nlu_intent_to_state(nlu_result.intent, state)
        state.current_intent = intent
        self._add_reasoning_step(reasoning, f"  - 最终意图: {intent.value}")
        
        state.missing_fields = missing_fields
        state.add_reasoning(self.name, "\n".join(reasoning))
        
        return state
    
    def _map_nlu_intent_to_state(self, nlu_intent: str, state: TripState) -> IntentType:
        if nlu_intent == "TRIP_REQUEST":
            return IntentType.INITIAL_REQUEST
        elif nlu_intent == "INFO_PROVIDED":
            return IntentType.INFO_PROVIDED
        elif nlu_intent == "CONFIRM":
            return IntentType.CONFIRM
        elif nlu_intent == "MODIFY_SELECTION":
            return IntentType.MODIFY_SELECTION
        else:
            has_destination = bool(state.destination)
            has_origin = bool(state.origin)
            if has_origin and has_destination:
                return IntentType.INFO_PROVIDED
            return IntentType.UNKNOWN
    
    def _identify_missing_fields(self, state: TripState) -> List[str]:
        required_fields = []
        
        if not state.user_level:
            required_fields.append("user_level")
        if not state.origin:
            required_fields.append("origin")
        if not state.destination:
            required_fields.append("destination")
        if not state.departure_time:
            required_fields.append("departure_time")
        if not state.return_time:
            required_fields.append("return_time")
        if not state.travel_mode:
            required_fields.append("travel_mode")
        if not state.hotel_type:
            required_fields.append("hotel_type")
        if not state.trip_purpose:
            required_fields.append("trip_purpose")
        if not state.customer_location:
            required_fields.append("customer_location")
        
        if state.has_banquet:
            if not state.banquet_budget:
                required_fields.append("banquet_budget")
            if not state.banquet_time:
                required_fields.append("banquet_time")
            if not state.banquet_location:
                required_fields.append("banquet_location")
        
        return required_fields
