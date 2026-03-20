from typing import Dict, Any, List
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, IntentType


class IntentClassificationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Intent-Classification-Agent")
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        user_input = kwargs.get("user_input", "")
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, "Step 1: 接收到用户输入，开始意图分类")
        self._add_reasoning_step(reasoning, f"用户输入: {user_input}")
        
        current_fields = state.get_all_fields()
        filled_fields = {k: v for k, v in current_fields.items() if v not in ("", None, 0, False)}
        
        self._add_reasoning_step(reasoning, f"Step 2: 当前已填写的字段: {list(filled_fields.keys())}")
        
        has_destination = bool(state.destination)
        has_origin = bool(state.origin)
        missing_fields = self._identify_missing_fields(state)
        
        self._add_reasoning_step(reasoning, f"Step 3: 识别缺失字段: {missing_fields}")
        
        if not has_origin and not has_destination:
            intent = IntentType.INITIAL_REQUEST
            self._add_reasoning_step(reasoning, "判断: 初次请求（无出发地和目的地）")
        elif has_origin and has_destination and len(missing_fields) > 0:
            intent = IntentType.INFO_PROVIDED
            self._add_reasoning_step(reasoning, "判断: 信息提供状态（已有行程但有缺失字段）")
        elif missing_fields:
            intent = IntentType.INFO_PROVIDED
            self._add_reasoning_step(reasoning, "判断: 追问回答")
        elif kwargs.get("user_confirm", False):
            intent = IntentType.CONFIRM
            self._add_reasoning_step(reasoning, "判断: 用户确认提交")
        else:
            intent = IntentType.UNKNOWN
            self._add_reasoning_step(reasoning, "判断: 无法识别的意图")
        
        self._add_reasoning_step(reasoning, f"Step 4: 最终意图分类结果: {intent.value}")
        
        state.current_intent = intent
        state.missing_fields = missing_fields
        state.add_reasoning(self.name, "\n".join(reasoning))
        
        return state
    
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
        elif state.banquet_budget > 0 or state.banquet_time:
            state.has_banquet = True
        
        return required_fields
