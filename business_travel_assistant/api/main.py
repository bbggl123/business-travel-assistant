from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from business_travel_assistant.workflow.trip_workflow import TripWorkflow
from business_travel_assistant.models.trip_state import TripState


app = FastAPI(title="Business Travel Assistant API")
workflow = TripWorkflow()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "user1"


class ChatResponse(BaseModel):
    message: str
    questions: List[str] = []
    state_summary: dict = {}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result_state = await workflow.run(request.message, request.user_id)
        
        response_message = ""
        questions = []
        
        if result_state.pending_questions:
            questions = [q.question_text for q in result_state.pending_questions[:3]]
            response_message = f"我需要确认一些信息来帮助您规划行程："
        elif result_state.transport_options:
            response_message = "已为您生成以下出行方案，请选择："
        elif result_state.workflow_status.value == "COMPLETED":
            response_message = "审批单已生成，具体内容如下："
        
        return ChatResponse(
            message=response_message,
            questions=questions,
            state_summary={
                "status": result_state.workflow_status.value,
                "intent": result_state.current_intent.value,
                "missing_fields": result_state.missing_fields,
                "transport_options_count": len(result_state.transport_options),
                "hotel_options_count": len(result_state.hotel_options),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
