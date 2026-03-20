from typing import List
from business_travel_assistant.agents.base_agent import BaseAgent
from business_travel_assistant.models.trip_state import TripState, Question


FIELD_QUESTION_MAP = {
    "user_level": {
        "question": "请问您的职级是？",
        "options": ["初级/基层员工", "中级/中层管理者", "高级/高层管理者"]
    },
    "trip_purpose": {
        "question": "请问此次出差的主要目的是？",
        "options": ["客户拜访", "项目实施", "商务谈判", "技术支持", "内部会议"]
    },
    "customer_location": {
        "question": "请问客户的具体地址是？",
        "options": None
    },
    "travel_mode": {
        "question": "请问您倾向于哪种出行方式？",
        "options": ["飞机", "火车/高铁", "自驾", "其他"]
    },
    "departure_time": {
        "question": "请问您计划什么时候出发？",
        "options": None
    },
    "return_time": {
        "question": "请问您计划什么时候返回？",
        "options": None
    },
    "hotel_type": {
        "question": "请问您需要什么类型的住宿？",
        "options": ["经济型", "舒适型", "高档/豪华型"]
    },
    "hotel_requirement": {
        "question": "请问您对酒店有什么特殊要求吗？",
        "options": None
    },
    "has_banquet": {
        "question": "此次出差是否涉及宴请客户？",
        "options": ["是", "否"]
    },
    "banquet_time": {
        "question": "请问宴请计划在什么时候？",
        "options": None
    },
    "banquet_location": {
        "question": "请问宴请地点在哪里？",
        "options": None
    },
    "banquet_diet": {
        "question": "请问客户有什么饮食要求吗？（如清真、素食等）",
        "options": None
    },
    "banquet_budget": {
        "question": "请问您计划宴请花费多少金额？",
        "options": None
    },
}


class InformationCollectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("Information-Collection-Agent")
    
    async def execute(self, state: TripState, **kwargs) -> TripState:
        reasoning = self._init_cot_reasoning()
        
        self._add_reasoning_step(reasoning, "Step 1: 开始信息收集，识别缺失字段")
        
        missing_fields = self._identify_missing_fields(state)
        self._add_reasoning_step(reasoning, f"识别的缺失字段: {missing_fields}")
        
        self._add_reasoning_step(reasoning, "Step 2: 生成追问问题列表")
        
        questions = []
        for missing_field in missing_fields:
            if missing_field in FIELD_QUESTION_MAP:
                config = FIELD_QUESTION_MAP[missing_field]
                q = Question(
                    field_name=missing_field,
                    question_text=config["question"],
                    options=config["options"],
                    allow_custom=config["options"] is None,
                    reasoning=[]
                )
                questions.append(q)
                self._add_reasoning_step(reasoning, f"  - {missing_field}: {config['question']}")
            else:
                q = Question(
                    field_name=missing_field,
                    question_text=f"请提供您的{missing_field}",
                    allow_custom=True,
                    reasoning=[]
                )
                questions.append(q)
        
        self._add_reasoning_step(reasoning, f"Step 3: 共生成 {len(questions)} 个追问问题")
        
        state.missing_fields = missing_fields
        state.pending_questions = questions
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
        
        return required_fields
