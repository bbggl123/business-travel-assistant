from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class IntentType(Enum):
    INITIAL_REQUEST = "INITIAL_REQUEST"
    INFO_PROVIDED = "INFO_PROVIDED"
    MODIFY_SELECTION = "MODIFY_SELECTION"
    CONFIRM = "CONFIRM"
    UNKNOWN = "UNKNOWN"


class WorkflowStatus(Enum):
    INITIAL = "INITIAL"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    COLLECTING_INFO = "COLLECTING_INFO"
    PLANNING = "PLANNING"
    BUDGET_ESTIMATION = "BUDGET_ESTIMATION"
    USER_CONFIRMATION = "USER_CONFIRMATION"
    APPROVAL_GENERATION = "APPROVAL_GENERATION"
    INVOICE_PROCESSING = "INVOICE_PROCESSING"
    COMPLETED = "COMPLETED"
    END = "END"


@dataclass
class Question:
    field_name: str
    question_text: str
    options: Optional[List[str]] = None
    allow_custom: bool = True
    reasoning: List[str] = field(default_factory=list)


@dataclass
class TransportOption:
    type: str
    provider: str
    flight_no: Optional[str] = None
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    departure_station: Optional[str] = None
    arrival_station: Optional[str] = None
    price: float = 0.0
    is_compliant: bool = True
    exceeds_amount: float = 0.0
    reasoning: List[str] = field(default_factory=list)


@dataclass
class HotelOption:
    name: str
    type: str
    address: str
    distance_to_customer: float = 0.0
    price_per_night: float = 0.0
    total_price: float = 0.0
    is_compliant: bool = True
    exceeds_amount: float = 0.0
    transportation: str = ""
    reasoning: List[str] = field(default_factory=list)


@dataclass
class RestaurantOption:
    name: str
    address: str
    cuisine_type: str
    distance_to_location: float = 0.0
    avg_price_per_person: float = 0.0
    max_budget: float = 0.0
    is_compliant: bool = True
    exceeds_amount: float = 0.0
    recommended_dishes: List[Dict[str, Any]] = field(default_factory=list)
    total_estimated_cost: float = 0.0
    reasoning: List[str] = field(default_factory=list)


@dataclass
class TripState:
    user_id: str = ""
    user_level: str = ""
    
    origin: str = ""
    destination: str = ""
    departure_time: Optional[datetime] = None
    return_time: Optional[datetime] = None
    travel_mode: str = ""
    
    hotel_type: str = ""
    hotel_requirement: str = ""
    
    trip_purpose: str = ""
    customer_location: str = ""
    
    has_banquet: bool = False
    banquet_time: Optional[datetime] = None
    banquet_location: str = ""
    banquet_diet: str = ""
    banquet_budget: float = 0.0
    
    missing_fields: List[str] = field(default_factory=list)
    pending_questions: List[Question] = field(default_factory=list)
    
    transport_options: List[TransportOption] = field(default_factory=list)
    hotel_options: List[HotelOption] = field(default_factory=list)
    restaurant_options: List[RestaurantOption] = field(default_factory=list)
    
    agent_reasoning: Dict[str, List[str]] = field(default_factory=dict)
    
    selected_transport: Optional[TransportOption] = None
    selected_hotel: Optional[HotelOption] = None
    selected_restaurant: Optional[RestaurantOption] = None
    
    current_intent: IntentType = IntentType.UNKNOWN
    workflow_status: WorkflowStatus = WorkflowStatus.INITIAL
    
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_reasoning(self, agent_name: str, step: str):
        if agent_name not in self.agent_reasoning:
            self.agent_reasoning[agent_name] = []
        self.agent_reasoning[agent_name].append(step)
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_all_fields(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_level": self.user_level,
            "origin": self.origin,
            "destination": self.destination,
            "departure_time": self.departure_time,
            "return_time": self.return_time,
            "travel_mode": self.travel_mode,
            "hotel_type": self.hotel_type,
            "hotel_requirement": self.hotel_requirement,
            "trip_purpose": self.trip_purpose,
            "customer_location": self.customer_location,
            "has_banquet": self.has_banquet,
            "banquet_time": self.banquet_time,
            "banquet_location": self.banquet_location,
            "banquet_diet": self.banquet_diet,
            "banquet_budget": self.banquet_budget,
        }
