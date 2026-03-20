import re
from typing import Dict, List, Any, Optional
from business_travel_assistant.services.nlu.base_nlu import BaseNLUService, SemanticParseResult, ExtractedEntity


LOCATION_PATTERNS = {
    r"北京": "北京",
    r"上海": "上海",
    r"广州": "广州",
    r"深圳": "深圳",
    r"杭州": "杭州",
    r"南京": "南京",
    r"成都": "成都",
    r"武汉": "武汉",
    r"西安": "西安",
    r"重庆": "重庆",
}

TIME_PATTERNS = {
    r"明天": 1,
    r"后天": 2,
    r"大后天": 3,
    r"今天": 0,
    r"下周一": 7,
    r"下周二": 8,
}

TIME_OF_DAY_PATTERNS = {
    r"早上|上午|早晨": "上午",
    r"中午|中午": "中午",
    r"下午|午后": "下午",
    r"晚上|傍晚|夜间": "晚上",
}

TRAVEL_MODE_PATTERNS = {
    r"飞机|航班|飞": "飞机",
    r"高铁|动车|火车|列车": "火车",
    r"自驾|开车|自己开车": "自驾",
}

HOTEL_TYPE_PATTERNS = {
    r"经济型|便宜|省钱": "经济型",
    r"舒适型|中等|普通": "舒适型",
    r"高档|豪华|五星级|四星级": "高档/豪华型",
}

USER_LEVEL_PATTERNS = {
    r"基层|初级|普通员工": "初级/基层员工",
    r"中层|经理|主管": "中级/中层管理者",
    r"高层|总监|总裁|总经理": "高级/高层管理者",
}

TRIP_PURPOSE_PATTERNS = {
    r"客户拜访|拜访客户|见客户": "客户拜访",
    r"项目实施|实施|部署": "项目实施",
    r"商务谈判|谈判|谈合作": "商务谈判",
    r"技术支持|技术支持|技术交流": "技术支持",
    r"内部会议|开会|会议": "内部会议",
}

BANQUET_PATTERNS = {
    r"宴请|请客|请客户吃饭": True,
    r"不需要宴请|不用宴请|没有宴请": False,
}


class MockNLUService(BaseNLUService):
    def __init__(self):
        super().__init__("Mock-NLU-Service")
    
    async def parse(self, text: str, context: Optional[Dict[str, Any]] = None) -> SemanticParseResult:
        reasoning = []
        
        reasoning.append(f"[MockNLU] Step 1: 接收文本输入: {text}")
        
        entities = await self.extract_entities(text)
        reasoning.append(f"[MockNLU] Step 2: 提取实体 {len(entities)} 个")
        
        intent_scores = await self.classify_intent(text)
        top_intent = max(intent_scores.items(), key=lambda x: x[1])
        reasoning.append(f"[MockNLU] Step 3: 意图分类结果: {top_intent[0]} (置信度: {top_intent[1]:.2f})")
        
        extracted_data = self._extract_structured_data(text, entities)
        reasoning.append(f"[MockNLU] Step 4: 结构化数据提取: {extracted_data}")
        
        reasoning.append(f"[MockNLU] Step 5: 语义解析完成")
        
        return SemanticParseResult(
            original_text=text,
            intent=top_intent[0],
            entities=entities,
            confidence=top_intent[1],
            reasoning_steps=reasoning,
            raw_output=extracted_data
        )
    
    async def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[ExtractedEntity]:
        entities = []
        
        for pattern, location in LOCATION_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="LOCATION",
                    value=location,
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, time_delta in TIME_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="TIME_RELATIVE",
                    value=str(time_delta),
                    confidence=0.90,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, time_desc in TIME_OF_DAY_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="TIME_OF_DAY",
                    value=time_desc,
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, mode in TRAVEL_MODE_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="TRAVEL_MODE",
                    value=mode,
                    confidence=0.90,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, hotel_type in HOTEL_TYPE_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="HOTEL_TYPE",
                    value=hotel_type,
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, level in USER_LEVEL_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="USER_LEVEL",
                    value=level,
                    confidence=0.90,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, purpose in TRIP_PURPOSE_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="TRIP_PURPOSE",
                    value=purpose,
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        for pattern, has_banquet in BANQUET_PATTERNS.items():
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    entity_type="BANQUET_REQUIRED",
                    value=str(has_banquet),
                    confidence=0.90,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        budget_match = re.search(r"(\d+)\s*(?:元|块钱|预算)", text)
        if budget_match:
            entities.append(ExtractedEntity(
                entity_type="BUDGET",
                value=budget_match.group(1),
                confidence=0.85,
                start_pos=budget_match.start(),
                end_pos=budget_match.end()
            ))
        
        date_match = re.search(r"(\d{1,2})[月\-\/](\d{1,2})[日号]?", text)
        if date_match:
            entities.append(ExtractedEntity(
                entity_type="DATE",
                value=f"{date_match.group(1)}-{date_match.group(2)}",
                confidence=0.80,
                start_pos=date_match.start(),
                end_pos=date_match.end()
            ))
        
        time_match = re.search(r"(\d{1,2})[时点\:](\d{2})", text)
        if time_match:
            entities.append(ExtractedEntity(
                entity_type="TIME_SPECIFIC",
                value=f"{time_match.group(1)}:{time_match.group(2)}",
                confidence=0.85,
                start_pos=time_match.start(),
                end_pos=time_match.end()
            ))
        
        return entities
    
    async def classify_intent(self, text: str, candidate_intents: Optional[List[str]] = None) -> Dict[str, float]:
        text_lower = text.lower()
        
        intent_scores = {
            "TRIP_REQUEST": 0.0,
            "INFO_PROVIDED": 0.0,
            "MODIFY_SELECTION": 0.0,
            "CONFIRM": 0.0,
            "ASK_QUESTION": 0.0,
            "UNKNOWN": 0.0
        }
        
        if any(keyword in text_lower for keyword in ["出差", "去", "到", "出发", "行程", "安排"]):
            intent_scores["TRIP_REQUEST"] += 0.4
        
        if any(keyword in text_lower for keyword in ["明天", "后天", "今天", "周一", "周二"]):
            intent_scores["TRIP_REQUEST"] += 0.2
        
        if any(keyword in text_lower for keyword in ["是", "对的", "正确", "确认"]):
            intent_scores["INFO_PROVIDED"] += 0.3
        
        if any(keyword in text_lower for keyword in ["选择", "选", "要", "决定"]):
            intent_scores["MODIFY_SELECTION"] += 0.3
        
        if any(keyword in text_lower for keyword in ["确认", "提交", "好的", "可以", "行"]):
            intent_scores["CONFIRM"] += 0.4
        
        if any(keyword in text_lower for keyword in ["？", "?", "怎么", "如何", "什么"]):
            intent_scores["ASK_QUESTION"] += 0.3
        
        max_score = max(intent_scores.values())
        if max_score < 0.1:
            intent_scores["UNKNOWN"] = 0.5
        else:
            for intent in intent_scores:
                if intent_scores[intent] == max_score:
                    intent_scores[intent] = min(max_score + 0.1, 1.0)
                    break
        
        return intent_scores
    
    def _extract_structured_data(self, text: str, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        result = {
            "origin": None,
            "destination": None,
            "departure_time": None,
            "return_time": None,
            "travel_mode": None,
            "hotel_type": None,
            "user_level": None,
            "trip_purpose": None,
            "has_banquet": None,
            "banquet_budget": None,
        }
        
        locations = [e.value for e in entities if e.entity_type == "LOCATION"]
        if len(locations) >= 2:
            result["origin"] = locations[0]
            result["destination"] = locations[1]
        elif len(locations) == 1:
            if "去" in text or "到" in text or "飞" in text:
                result["destination"] = locations[0]
        
        for entity in entities:
            if entity.entity_type == "TRAVEL_MODE":
                result["travel_mode"] = entity.value
            elif entity.entity_type == "HOTEL_TYPE":
                result["hotel_type"] = entity.value
            elif entity.entity_type == "USER_LEVEL":
                result["user_level"] = entity.value
            elif entity.entity_type == "TRIP_PURPOSE":
                result["trip_purpose"] = entity.value
            elif entity.entity_type == "BANQUET_REQUIRED":
                result["has_banquet"] = entity.value == "True"
            elif entity.entity_type == "BUDGET":
                result["banquet_budget"] = float(entity.value)
        
        return result
