from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, TransportOption, HotelOption, RestaurantOption


@dataclass
class ApprovalForm:
    form_id: str
    applicant_name: str
    applicant_level: str
    trip_purpose: str
    
    origin: str
    destination: str
    departure_time: Optional[datetime] = None
    return_time: Optional[datetime] = None
    
    selected_transport: Optional[Dict[str, Any]] = None
    transport_total: float = 0.0
    
    selected_hotel: Optional[Dict[str, Any]] = None
    hotel_nights: int = 1
    hotel_total: float = 0.0
    
    has_banquet: bool = False
    selected_restaurant: Optional[Dict[str, Any]] = None
    banquet_estimated_cost: float = 0.0
    
    subtotal: float = 0.0
    total_exceeds: float = 0.0
    items_exceeding_standard: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=datetime.now)
    approval_url: Optional[str] = None


class ApprovalGenerationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Approval-Generation-Agent")
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, "Step 1: 开始生成审批单")
        self._add_reasoning_step(reasoning, f"申请人: {state.user_id}, 职级: {state.user_level}")
        
        self._add_reasoning_step(reasoning, "Step 2: 汇总各项费用")
        
        transport_total = state.selected_transport.price if state.selected_transport else 0
        hotel_total = state.selected_hotel.total_price if state.selected_hotel else 0
        banquet_total = state.selected_restaurant.total_estimated_cost if state.selected_restaurant else 0
        
        self._add_reasoning_step(reasoning, f"交通费用: {transport_total}")
        self._add_reasoning_step(reasoning, f"住宿费用: {hotel_total}")
        self._add_reasoning_step(reasoning, f"宴请费用: {banquet_total}")
        
        subtotal = transport_total + hotel_total + banquet_total
        self._add_reasoning_step(reasoning, f"费用合计: {subtotal}")
        
        self._add_reasoning_step(reasoning, "Step 3: 识别超标项目")
        
        exceeds_items = []
        total_exceeds = 0
        
        if state.selected_transport and state.selected_transport.exceeds_amount > 0:
            exceeds_items.append(f"交通超标 {state.selected_transport.exceeds_amount}元")
            total_exceeds += state.selected_transport.exceeds_amount
        
        if state.selected_hotel and state.selected_hotel.exceeds_amount > 0:
            exceeds_items.append(f"住宿超标 {state.selected_hotel.exceeds_amount}元")
            total_exceeds += state.selected_hotel.exceeds_amount
        
        if state.selected_restaurant and state.selected_restaurant.exceeds_amount > 0:
            exceeds_items.append(f"宴请超标 {state.selected_restaurant.exceeds_amount}元")
            total_exceeds += state.selected_restaurant.exceeds_amount
        
        self._add_reasoning_step(reasoning, f"超标项目: {exceeds_items}")
        self._add_reasoning_step(reasoning, f"总超标金额: {total_exceeds}")
        
        self._add_reasoning_step(reasoning, "Step 4: 生成审批单")
        
        form_id = f"TRIP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        approval_form = ApprovalForm(
            form_id=form_id,
            applicant_name=state.user_id,
            applicant_level=state.user_level,
            trip_purpose=state.trip_purpose,
            origin=state.origin,
            destination=state.destination,
            departure_time=state.departure_time,
            return_time=state.return_time,
            selected_transport={
                "type": state.selected_transport.type if state.selected_transport else "",
                "provider": state.selected_transport.provider if state.selected_transport else "",
                "flight_no": state.selected_transport.flight_no if state.selected_transport else "",
                "price": state.selected_transport.price if state.selected_transport else 0,
            } if state.selected_transport else None,
            transport_total=transport_total,
            selected_hotel={
                "name": state.selected_hotel.name if state.selected_hotel else "",
                "type": state.selected_hotel.type if state.selected_hotel else "",
                "address": state.selected_hotel.address if state.selected_hotel else "",
                "price_per_night": state.selected_hotel.price_per_night if state.selected_hotel else 0,
            } if state.selected_hotel else None,
            hotel_nights=1,
            hotel_total=hotel_total,
            has_banquet=state.has_banquet,
            selected_restaurant={
                "name": state.selected_restaurant.name if state.selected_restaurant else "",
                "cuisine_type": state.selected_restaurant.cuisine_type if state.selected_restaurant else "",
                "estimated_cost": state.selected_restaurant.total_estimated_cost if state.selected_restaurant else 0,
            } if state.selected_restaurant else None,
            banquet_estimated_cost=banquet_total,
            subtotal=subtotal,
            total_exceeds=total_exceeds,
            items_exceeding_standard=exceeds_items,
            generated_at=datetime.now(),
            approval_url=f"https://approval.example.com/{form_id}"
        )
        
        self._add_reasoning_step(reasoning, f"审批单ID: {form_id}")
        self._add_reasoning_step(reasoning, "Step 5: 审批单生成完成")
        
        state.add_reasoning(self.name, "\n".join(reasoning))
        state.add_message("assistant", f"审批单已生成，单号: {form_id}")
        
        return state
    
    def export_form_text(self, form: ApprovalForm) -> str:
        lines = [
            "=" * 50,
            "商旅出差审批单",
            "=" * 50,
            f"审批单号: {form.form_id}",
            f"申请人: {form.applicant_name}",
            f"职级: {form.applicant_level}",
            f"出差目的: {form.trip_purpose}",
            "",
            "行程信息:",
            f"出发地: {form.origin}",
            f"目的地: {form.destination}",
            f"出发时间: {form.departure_time.strftime('%Y-%m-%d %H:%M') if form.departure_time else '未填写'}",
            f"返程时间: {form.return_time.strftime('%Y-%m-%d %H:%M') if form.return_time else '未填写'}",
            "",
            "交通方案:",
        ]
        
        if form.selected_transport:
            lines.append(f"  {form.selected_transport['type']}: {form.selected_transport['provider']} {form.selected_transport['flight_no']}")
            lines.append(f"  费用: {form.transport_total}元")
        else:
            lines.append("  未选择")
        
        lines.extend([
            "",
            "住宿方案:",
        ])
        
        if form.selected_hotel:
            lines.append(f"  {form.selected_hotel['name']} ({form.selected_hotel['type']})")
            lines.append(f"  地址: {form.selected_hotel['address']}")
            lines.append(f"  每晚: {form.selected_hotel['price_per_night']}元 x {form.hotel_nights}晚")
            lines.append(f"  费用: {form.hotel_total}元")
        else:
            lines.append("  未选择")
        
        lines.extend([
            "",
            "宴请安排:",
        ])
        
        if form.has_banquet and form.selected_restaurant:
            lines.append(f"  餐厅: {form.selected_restaurant['name']}")
            lines.append(f"  菜系: {form.selected_restaurant['cuisine_type']}")
            lines.append(f"  预计花费: {form.banquet_estimated_cost}元")
        elif form.has_banquet:
            lines.append("  需要宴请（未选择餐厅）")
        else:
            lines.append("  无宴请安排")
        
        lines.extend([
            "",
            "=" * 50,
            f"费用合计: {form.subtotal}元",
        ])
        
        if form.items_exceeding_standard:
            lines.append(f"超标金额: {form.total_exceeds}元")
            lines.append("超标项目:")
            for item in form.items_exceeding_standard:
                lines.append(f"  - {item}")
        
        lines.append("=" * 50)
        lines.append(f"审批链接: {form.approval_url}")
        
        return "\n".join(lines)
