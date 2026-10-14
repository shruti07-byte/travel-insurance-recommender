from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from groq import Groq
from functions import get_recommendation, validate_input, chatbot_reply

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://travel-insurance-recommender.vercel.app",
        "https://travel-insurance-recommender-git-main.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TravelData(BaseModel):
    destination: str
    trip_cost: int
    duration: int
    age: int
    medical_condition: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/")
def home():
    return {"message": "Travel Insurance Backend Running"}


@app.post("/recommend")
def recommend(data: TravelData):
    validation = validate_input(
        destination=data.destination,
        trip_cost=data.trip_cost,
        duration=data.duration,
        age=data.age,
        medical_condition=data.medical_condition
    )

    if validation["status"] == "error":
        return {"error": validation["errors"]}

    recommendation = get_recommendation(
        client=client,
        destination=data.destination,
        trip_cost=data.trip_cost,
        duration=data.duration,
        age=data.age,
        medical_condition=data.medical_condition
    )
    return {"recommendation": recommendation}


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = chatbot_reply(client, messages)

    if result["done"] and result["data"]:
        d = result["data"]
        try:
            recommendation = get_recommendation(
                client=client,
                destination=str(d["destination"]),
                trip_cost=int(d["trip_cost"]),
                duration=int(d["duration"]),
                age=int(d["age"]),
                medical_condition=str(d["medical_condition"])
            )
            return {"reply": recommendation, "done": True}
        except Exception as e:
            return {"reply": f"Sorry, something went wrong: {str(e)}", "done": False}

    return {"reply": result["reply"], "done": False}