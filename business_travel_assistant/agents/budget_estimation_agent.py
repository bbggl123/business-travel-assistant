from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, TransportOption, HotelOption, RestaurantOption


@dataclass
class BudgetEstimation:
    transport_total: float
    hotel_total: float
    banquet_total: float
    grand_total: float
    
    transport_standard: float
    hotel_standard: float
    banquet_standard: float
    total_standard: float
    
    exceeds_amount: float
    exceeds_items: List[str]
    
    risk_level: str
    
    reasoning: List[str]


class BudgetEstimationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Budget-Estimation-Agent")
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, "Step 1: 开始预算估算")
        
        selected_transport = state.selected_transport
        selected_hotel = state.selected_hotel
        selected_restaurant = state.selected_restaurant
        
        transport_total = selected_transport.price if selected_transport else 0
        hotel_total = selected_hotel.total_price if selected_hotel else 0
        banquet_total = selected_restaurant.total_estimated_cost if selected_restaurant else 0
        
        self._add_reasoning_step(reasoning, f"交通费用: {transport_total}")
        self._add_reasoning_step(reasoning, f"住宿费用: {hotel_total}")
        self._add_reasoning_step(reasoning, f"宴请费用: {banquet_total}")
        
        grand_total = transport_total + hotel_total + banquet_total
        self._add_reasoning_step(reasoning, f"预估总费用: {grand_total}")
        
        self._add_reasoning_step(reasoning, "Step 2: 对比差旅标准")
        
        standards = self._get_standards(state.user_level)
        transport_standard = standards["transport"]
        hotel_standard = standards["hotel"]
        banquet_standard = standards["banquet"]
        total_standard = transport_standard + hotel_standard + banquet_standard
        
        self._add_reasoning_step(reasoning, f"交通标准: {transport_standard}")
        self._add_reasoning_step(reasoning, f"住宿标准: {hotel_standard}")
        self._add_reasoning_step(reasoning, f"宴请标准: {banquet_standard}")
        self._add_reasoning_step(reasoning, f"总标准: {total_standard}")
        
        exceeds_items = []
        exceeds_amount = 0
        
        if selected_transport and selected_transport.exceeds_amount > 0:
            exceeds_items.append(f"交通超标: {selected_transport.exceeds_amount}元")
            exceeds_amount += selected_transport.exceeds_amount
        
        if selected_hotel and selected_hotel.exceeds_amount > 0:
            exceeds_items.append(f"住宿超标: {selected_hotel.exceeds_amount}元")
            exceeds_amount += selected_hotel.exceeds_amount
        
        if selected_restaurant and selected_restaurant.exceeds_amount > 0:
            exceeds_items.append(f"宴请超标: {selected_restaurant.exceeds_amount}元")
            exceeds_amount += selected_restaurant.exceeds_amount
        
        risk_level = "HIGH" if exceeds_amount > 500 else "MEDIUM" if exceeds_amount > 0 else "LOW"
        
        self._add_reasoning_step(reasoning, f"超标金额: {exceeds_amount}")
        self._add_reasoning_step(reasoning, f"超标项目: {exceeds_items}")
        self._add_reasoning_step(reasoning, f"风险等级: {risk_level}")
        
        state.add_reasoning(self.name, "\n".join(reasoning))
        
        return state
    
    def _get_standards(self, user_level: str) -> Dict[str, float]:
        standards_map = {
            "初级/基层员工": {"transport": 500, "hotel": 300, "banquet": 150},
            "中级/中层管理者": {"transport": 800, "hotel": 500, "banquet": 300},
            "高级/高层管理者": {"transport": 1500, "hotel": 800, "banquet": 500},
        }
        return standards_map.get(user_level, standards_map["初级/基层员工"])
