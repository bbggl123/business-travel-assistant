from typing import List, Dict, Any
from datetime import datetime
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, TransportOption, HotelOption, RestaurantOption
from business_travel_assistant.agents.compliance_check_agent import ComplianceCheckAgent


MOCK_TRANSPORT_DATA = {
    "flights": [
        {"flight_no": "CA1234", "airline": "中国国航", "origin": "北京", "destination": "上海",
         "departure_time": "08:00", "arrival_time": "10:30", "price": 680},
        {"flight_no": "MU5678", "airline": "东方航空", "origin": "北京", "destination": "上海",
         "departure_time": "09:30", "arrival_time": "12:00", "price": 720},
        {"flight_no": "CZ9012", "airline": "南方航空", "origin": "北京", "destination": "上海",
         "departure_time": "14:00", "arrival_time": "16:30", "price": 850},
        {"flight_no": "HU3456", "airline": "海南航空", "origin": "北京", "destination": "上海",
         "departure_time": "18:00", "arrival_time": "20:30", "price": 590},
    ],
    "trains": [
        {"train_no": "G1234", "type": "高铁", "origin": "北京", "destination": "上海",
         "departure_time": "08:00", "arrival_time": "13:00", "price": 553},
        {"train_no": "G5678", "type": "高铁", "origin": "北京", "destination": "上海",
         "departure_time": "10:00", "arrival_time": "15:00", "price": 553},
        {"train_no": "D2345", "type": "动车", "origin": "北京", "destination": "上海",
         "departure_time": "12:00", "arrival_time": "18:00", "price": 450},
    ]
}

MOCK_HOTEL_DATA = {
    "上海": {
        "高档/豪华型": [
            {"name": "上海外滩豪华酒店", "address": "上海浦东新区外滩路100号", "star_rating": 5, "price_per_night": 1200, "distance_to_center": 2.5},
            {"name": "上海金茂君悦大酒店", "address": "上海浦东新区世纪大道88号", "star_rating": 5, "price_per_night": 1100, "distance_to_center": 3.0},
            {"name": "上海静安香格里拉大酒店", "address": "上海静安区延安中路1218号", "star_rating": 5, "price_per_night": 1300, "distance_to_center": 4.0},
        ],
        "舒适型": [
            {"name": "上海中亚饭店", "address": "上海静安区恒丰路555号", "star_rating": 4, "price_per_night": 450, "distance_to_center": 1.5},
            {"name": "上海锦江之星", "address": "上海黄浦区南京东路680号", "star_rating": 3, "price_per_night": 280, "distance_to_center": 2.0},
        ],
        "经济型": [
            {"name": "如家快捷酒店", "address": "上海黄浦区西藏南路500号", "star_rating": 2, "price_per_night": 180, "distance_to_center": 3.0},
            {"name": "汉庭酒店", "address": "上海静安区新闸路1000号", "star_rating": 2, "price_per_night": 160, "distance_to_center": 2.5},
        ]
    }
}

MOCK_RESTAURANT_DATA = [
    {"name": "老上海本帮菜", "cuisine_type": "本帮菜", "address": "上海浦东新区南京东路200号", "price_range": "150-300/人", "suitable_for": ["商务宴请"]},
    {"name": "粤港茶餐厅", "cuisine_type": "粤菜", "address": "上海浦东新区陆家嘴环路1000号", "price_range": "100-200/人", "suitable_for": ["商务宴请", "朋友聚餐"]},
    {"name": "鼎泰丰", "cuisine_type": "台菜", "address": "上海静安区南京西路1266号", "price_range": "150-250/人", "suitable_for": ["商务宴请"]},
    {"name": "沈大成酒楼", "cuisine_type": "本帮菜", "address": "上海黄浦区南京东路328号", "price_range": "80-150/人", "suitable_for": ["朋友聚餐"]},
]


class TripPlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__("Trip-Planning-Agent")
        self.compliance_agent = ComplianceCheckAgent()
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, "Step 1: 开始行程规划")
        self._add_reasoning_step(reasoning, f"目的地: {state.destination}, 出发地: {state.origin}")
        self._add_reasoning_step(reasoning, f"出行方式: {state.travel_mode}, 住宿类型: {state.hotel_type}")
        
        self._add_reasoning_step(reasoning, "Step 2: 规划交通方案")
        transport_options = await self._plan_transport(state)
        self._add_reasoning_step(reasoning, f"找到 {len(transport_options)} 个交通方案")
        
        self._add_reasoning_step(reasoning, "Step 3: 规划住宿方案")
        hotel_options = await self._plan_hotel(state)
        self._add_reasoning_step(reasoning, f"找到 {len(hotel_options)} 个住宿方案")
        
        self._add_reasoning_step(reasoning, "Step 4: 规划宴请方案")
        restaurant_options = await self._plan_restaurant(state)
        self._add_reasoning_step(reasoning, f"找到 {len(restaurant_options)} 个餐厅方案")
        
        state.transport_options = transport_options
        state.hotel_options = hotel_options
        state.restaurant_options = restaurant_options
        state.add_reasoning(self.name, "\n".join(reasoning))
        
        return state
    
    async def _plan_transport(self, state: TripState) -> List[TransportOption]:
        options = []
        
        travel_mode = state.travel_mode.lower() if state.travel_mode else ""
        
        if "飞机" in travel_mode or "flight" in travel_mode or not travel_mode:
            flights = MOCK_TRANSPORT_DATA.get("flights", [])
            for flight in flights:
                if flight["origin"] == state.origin and flight["destination"] == state.destination:
                    check_result = await self.compliance_agent.check_transport("飞机", flight["price"], state.user_level)
                    options.append(TransportOption(
                        type="飞机",
                        provider=flight["airline"],
                        flight_no=flight["flight_no"],
                        departure_time=datetime.strptime(flight["departure_time"], "%H:%M"),
                        arrival_time=datetime.strptime(flight["arrival_time"], "%H:%M"),
                        departure_station=state.origin,
                        arrival_station=state.destination,
                        price=flight["price"],
                        is_compliant=check_result["is_compliant"],
                        exceeds_amount=check_result["exceeds_amount"],
                        reasoning=[check_result["reasoning"]]
                    ))
        
        if "火车" in travel_mode or "高铁" in travel_mode or "train" in travel_mode or not travel_mode:
            trains = MOCK_TRANSPORT_DATA.get("trains", [])
            for train in trains:
                if train["origin"] == state.origin and train["destination"] == state.destination:
                    check_result = await self.compliance_agent.check_transport("火车", train["price"], state.user_level)
                    options.append(TransportOption(
                        type=train["type"],
                        provider="中国铁路",
                        flight_no=train["train_no"],
                        departure_time=datetime.strptime(train["departure_time"], "%H:%M"),
                        arrival_time=datetime.strptime(train["arrival_time"], "%H:%M"),
                        departure_station=state.origin,
                        arrival_station=state.destination,
                        price=train["price"],
                        is_compliant=check_result["is_compliant"],
                        exceeds_amount=check_result["exceeds_amount"],
                        reasoning=[check_result["reasoning"]]
                    ))
        
        return sorted(options, key=lambda x: x.price)
    
    async def _plan_hotel(self, state: TripState) -> List[HotelOption]:
        options = []
        
        city = state.destination.replace("市", "").replace("区", "").replace("县", "")
        hotel_data = MOCK_HOTEL_DATA.get(city, MOCK_HOTEL_DATA.get("上海", {}))
        
        hotel_type = state.hotel_type or "舒适型"
        
        for htype in [hotel_type, "舒适型", "经济型"]:
            hotels = hotel_data.get(htype, [])
            for hotel in hotels:
                check_result = await self.compliance_agent.check_hotel(htype, hotel["price_per_night"], state.user_level)
                options.append(HotelOption(
                    name=hotel["name"],
                    type=htype,
                    address=hotel["address"],
                    distance_to_customer=hotel["distance_to_center"],
                    price_per_night=hotel["price_per_night"],
                    total_price=hotel["price_per_night"],
                    is_compliant=check_result["is_compliant"],
                    exceeds_amount=check_result["exceeds_amount"],
                    transportation=f"地铁{hotel['distance_to_center']*10:.0f}分钟/出租车约{hotel['distance_to_center']*5:.0f}分钟",
                    reasoning=[check_result["reasoning"]]
                ))
        
        return sorted(options, key=lambda x: x.distance_to_customer)
    
    async def _plan_restaurant(self, state: TripState) -> List[RestaurantOption]:
        if not state.has_banquet:
            return []
        
        options = []
        
        for restaurant in MOCK_RESTAURANT_DATA:
            budget = state.banquet_budget or 150
            check_result = await self.compliance_agent.check_banquet(budget, state.user_level)
            
            options.append(RestaurantOption(
                name=restaurant["name"],
                address=restaurant["address"],
                cuisine_type=restaurant["cuisine_type"],
                distance_to_location=2.0,
                avg_price_per_person=150,
                max_budget=budget,
                is_compliant=check_result["is_compliant"],
                exceeds_amount=check_result["exceeds_amount"],
                recommended_dishes=[
                    {"name": "招牌菜A", "price": 80, "category": "主菜"},
                    {"name": "招牌菜B", "price": 60, "category": "配菜"},
                ],
                total_estimated_cost=budget,
                reasoning=[check_result["reasoning"]]
            ))
        
        return options
