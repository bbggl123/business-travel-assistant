from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TravelStandard:
    user_level: str
    
    flight_max_price: float = 1000
    train_max_price: float = 500
    
    hotel_economy_max: float = 200
    hotel_comfort_max: float = 300
    hotel_luxury_max: float = 500
    
    banquet_max_per_person: float = 150
    
    meal_allowance: float = 100
    taxi_allowance: float = 50


TRAVEL_STANDARDS: Dict[str, TravelStandard] = {
    "初级/基层员工": TravelStandard(
        user_level="初级/基层员工",
        flight_max_price=1000,
        train_max_price=500,
        hotel_economy_max=200,
        hotel_comfort_max=300,
        hotel_luxury_max=500,
        banquet_max_per_person=150,
    ),
    "中级/中层管理者": TravelStandard(
        user_level="中级/中层管理者",
        flight_max_price=1500,
        train_max_price=800,
        hotel_economy_max=300,
        hotel_comfort_max=500,
        hotel_luxury_max=800,
        banquet_max_per_person=300,
    ),
    "高级/高层管理者": TravelStandard(
        user_level="高级/高层管理者",
        flight_max_price=3000,
        train_max_price=1500,
        hotel_economy_max=500,
        hotel_comfort_max=800,
        hotel_luxury_max=1500,
        banquet_max_per_person=500,
    ),
}


class ComplianceCheckAgent:
    def __init__(self):
        self.name = "Compliance-Check-Agent"
    
    async def check_transport(self, transport_type: str, price: float, user_level: str) -> Dict:
        reasoning = []
        reasoning.append(f"[{self.name}] Step 1: 开始交通合规检查")
        reasoning.append(f"交通类型: {transport_type}, 价格: {price}, 职级: {user_level}")
        
        standard = self.get_standard(user_level)
        
        if transport_type in ["飞机", "flight"]:
            max_price = standard.flight_max_price
            reasoning.append(f"机票上限: {max_price}")
        else:
            max_price = standard.train_max_price
            reasoning.append(f"火车票上限: {max_price}")
        
        exceeds = max(0, price - max_price)
        is_compliant = price <= max_price
        
        reasoning.append(f"是否合规: {is_compliant}, 超标金额: {exceeds}")
        
        return {
            "is_compliant": is_compliant,
            "standard_limit": max_price,
            "actual_amount": price,
            "exceeds_amount": exceeds,
            "reasoning": "\n".join(reasoning)
        }
    
    async def check_hotel(self, hotel_type: str, price_per_night: float, user_level: str) -> Dict:
        reasoning = []
        reasoning.append(f"[{self.name}] Step 1: 开始住宿合规检查")
        reasoning.append(f"住宿类型: {hotel_type}, 每晚价格: {price_per_night}, 职级: {user_level}")
        
        standard = self.get_standard(user_level)
        
        if hotel_type in ["经济型"]:
            max_price = standard.hotel_economy_max
        elif hotel_type in ["舒适型"]:
            max_price = standard.hotel_comfort_max
        else:
            max_price = standard.hotel_luxury_max
        
        reasoning.append(f"住宿上限: {max_price}")
        
        exceeds = max(0, price_per_night - max_price)
        is_compliant = price_per_night <= max_price
        
        reasoning.append(f"是否合规: {is_compliant}, 超标金额: {exceeds}")
        
        return {
            "is_compliant": is_compliant,
            "standard_limit": max_price,
            "actual_amount": price_per_night,
            "exceeds_amount": exceeds,
            "reasoning": "\n".join(reasoning)
        }
    
    async def check_banquet(self, budget: float, user_level: str) -> Dict:
        reasoning = []
        reasoning.append(f"[{self.name}] Step 1: 开始宴请合规检查")
        reasoning.append(f"计划花费: {budget}, 职级: {user_level}")
        
        standard = self.get_standard(user_level)
        max_price = standard.banquet_max_per_person
        
        reasoning.append(f"宴请人均上限: {max_price}")
        
        exceeds = max(0, budget - max_price)
        is_compliant = budget <= max_price
        
        reasoning.append(f"是否合规: {is_compliant}, 超标金额: {exceeds}")
        
        return {
            "is_compliant": is_compliant,
            "standard_limit": max_price,
            "actual_amount": budget,
            "exceeds_amount": exceeds,
            "reasoning": "\n".join(reasoning)
        }
    
    def get_standard(self, user_level: str) -> TravelStandard:
        return TRAVEL_STANDARDS.get(user_level, TRAVEL_STANDARDS["初级/基层员工"])
