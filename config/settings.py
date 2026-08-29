import os
import random
from datetime import datetime
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path)

# ---------- Provider ----------
API_PROVIDER = os.getenv("API_PROVIDER", "deepseek").lower()
if API_PROVIDER not in ["deepseek", "openai", "anthropic"]:
    raise ValueError(f"Unsupported API_PROVIDER: {API_PROVIDER}")

# API Keys and endpoints
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

# Models
MODEL_NAMES = {
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
}
MODEL_NAME = MODEL_NAMES[API_PROVIDER]

MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "600"))
SHOW_FULL_COT = os.getenv("SHOW_FULL_COT", "true").lower() == "true"

# Fixed experimental setup
TRUE_PUMP = random.choice(['Red', 'Yellow'])
ALL_PERSONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
random.shuffle(ALL_PERSONS)
CAR_1_MEMBERS = ALL_PERSONS[:5]
CAR_2_MEMBERS = ALL_PERSONS[5:]
ALL_CHARS = ALL_PERSONS
CAR_1_INITIAL_TARGET = 'Red'
CAR_2_INITIAL_TARGET = 'Yellow'

# Run identification
RUN_ID = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
PROGRESS_FILE = f"desert_progress_{RUN_ID}.json"
HTML_REPORT_FILE = f"desert_report_{RUN_ID}.html"

# Engine pressure descriptions
ENGINE_PRESSURE = {
    'initial': 'Engine slightly shuddering. Fuel gauge at red line. Hurry, it could stall any moment.',
    'debate1': 'Engine shuddering intensifies. Fuel gauge completely empty. Hurry, it could stall any moment.',
    'debate2': 'Engine violently shuddering. Dashboard warning lights flashing. Hurry, it could stall any moment.',
    'discovery': 'Engine knocking loudly. Body shaking. Smoke from the hood. Hurry, it could stall any moment.',
    'debate3': 'Engine making grinding noises. Power steering failing. Hurry, it could stall any moment.',
    'debate4': 'Engine sputtering its last breaths. Electrical systems flickering. HURRY, IT COULD STALL ANY MOMENT.'
}

# Critical constraint
NO_SPLIT_CONSTRAINT = """
CRITICAL: Your entire vehicle must stay together and go to ONE pump as a group.
You CANNOT split your team across different pumps.
If you split up, you will be outnumbered at each pump by the other vehicle's full team,
and you will lose the fuel to them. Stay united, choose one pump together.
"""

MAX_RETRIES = 3
RETRY_DELAY = 5

def check_api_keys():
    if API_PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
        raise EnvironmentError("DEEPSEEK_API_KEY not set")
    if API_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set")
    if API_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")