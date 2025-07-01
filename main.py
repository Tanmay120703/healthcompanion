from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, computed_field, ValidationError
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import os

load_dotenv()  # Load the .env file

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# FastAPI app setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# Pydantic Model
# ---------------------------
class FitnessFormData(BaseModel):
    name: str
    age: int
    gender: str
    height: float  # in cm
    weight: float  # in kg
    city: str
    goal: str

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / ((self.height / 100) ** 2), 2)

    @computed_field
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return "Underweight"
        elif 18.5 <= self.bmi < 25:
            return "Normal weight"
        elif 25 <= self.bmi < 30:
            return "Overweight"
        else:
            return "Obese"

# ---------------------------
# OpenAI Integration
# ---------------------------
def build_prompt(question, goal=None, ingredients=None):
    base = f"""
You are a helpful Indian health assistant with expertise in:
- Diet & Nutrition
- Fitness & Workout Planning

User goal: {goal or 'General health'}
Available ingredients: {', '.join(ingredients) if ingredients else 'N/A'}

Question: {question}

Provide clear, relevant, and practical advice.
"""
    return base.strip()

client = OpenAI()

def ask_openai_model(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ---------------------------
# Routes
# ---------------------------

@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/fitness-goal", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit")
async def handle_form(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    city: str = Form(...),
    goal: str = Form(...)
):
    try:
        form_data = FitnessFormData(
            name=name,
            age=age,
            gender=gender,
            height=height,
            weight=weight,
            city=city,
            goal=goal
        )
    except ValidationError as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Invalid input: " + str(e)
        })

    # Macronutrient Calculation
    if form_data.gender.lower() == 'male':
        bmr = 10 * form_data.weight + 6.25 * form_data.height - 5 * form_data.age + 5
    else:
        bmr = 10 * form_data.weight + 6.25 * form_data.height - 5 * form_data.age - 161

    activity_factor = 1.55
    calories = bmr * activity_factor

    if form_data.goal == "lose":
        calories -= 500
    elif form_data.goal == "gain":
        calories += 500

    protein_g = form_data.weight * 2.2
    fat_g = form_data.weight * 1
    carbs_g = (calories - (protein_g * 4 + fat_g * 9)) / 4

    # Micronutrient Recommendations
    calcium = 1000 if form_data.age < 50 else 1200
    iron = 18 if form_data.gender == "female" else 8
    vitamin_c = 75 if form_data.gender == "female" else 90
    vitamin_d = 600 if form_data.age < 70 else 800
    vitamin_b12 = 2.4
    fiber = 25 if form_data.gender == "female" else 38
    magnesium = 320 if form_data.gender == "female" else 420
    potassium = 2600 if form_data.gender == "female" else 3400

    result = {
        "name": form_data.name,
        "city": form_data.city,
        "age": form_data.age,
        "gender": form_data.gender,
        "height": form_data.height,
        "weight": form_data.weight,
        "bmi": form_data.bmi,
        "verdict": form_data.verdict,
        "goal": form_data.goal,
        "advice": "Stay active and eat well!",
        "macros": {
            "calories": round(calories),
            "protein_g": round(protein_g),
            "carbs_g": round(carbs_g),
            "fat_g": round(fat_g)
        },
        "micros": {
            "calcium": calcium,
            "iron": iron,
            "vitamin_c": vitamin_c,
            "vitamin_d": vitamin_d,
            "vitamin_b12": vitamin_b12,
            "fiber": fiber,
            "magnesium": magnesium,
            "potassium": potassium
        }
    }

    request.session['result'] = result
    return RedirectResponse(url="/result", status_code=303)

@app.get("/result", response_class=HTMLResponse)
def show_result(request: Request):
    result = request.session.get("result")
    if not result:
        return RedirectResponse(url="/")
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

@app.get("/weight-loss", response_class=HTMLResponse)
def weight_loss_page(request: Request):
    return templates.TemplateResponse("weight_loss.html", {"request": request})

@app.get("/weight-gain", response_class=HTMLResponse)
def weight_gain_page(request: Request):
    return templates.TemplateResponse("weight_gain.html", {"request": request})

@app.get("/muscle-gain", response_class=HTMLResponse)
def muscle_gain_page(request: Request):
    return templates.TemplateResponse("muscle_gain.html", {"request": request})

@app.post("/ask-assistant")
async def ask_assistant(
    question: str = Body(...),
    goal: str = Body(default="General health"),
    ingredients: list[str] = Body(default=[])
):
    prompt = build_prompt(question, goal, ingredients)
    answer = ask_openai_model(prompt)
    return JSONResponse({"response": answer})


# Serve static files like images (QR code etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up HTML templates
templates = Jinja2Templates(directory="templates")

# WhatsApp number
WHATSAPP_NUMBER = "9967389409"  # Replace with your number



from fastapi import Form
from fastapi.responses import RedirectResponse

@app.get("/enroll", response_class=HTMLResponse)
async def show_enroll_form(request: Request, plan: str = "basic"):
    return templates.TemplateResponse("enroll.html", {"request": request, "plan": plan})

@app.post("/enroll")
async def handle_enroll(
    name: str = Form(...),
    email: str = Form(...),
    plan: str = Form(...)
):
    message = f"Hello, I have enrolled.\nName: {name}\nEmail: {email}\nPlan: ₹{plan}"
    encoded_message = message.replace(' ', '%20').replace('\n', '%0A')
    whatsapp_url = f"https://wa.me/{WHATSAPP_NUMBER}?text={encoded_message}"
    return RedirectResponse(url=whatsapp_url, status_code=303)

