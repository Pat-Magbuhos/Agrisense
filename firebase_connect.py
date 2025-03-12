import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

# Load environment variables from the correct .env location
dotenv_path = os.path.join(os.path.dirname(__file__), "venv", ".env")  # Make sure to include "venv"
load_dotenv(dotenv_path)

# Retrieve Firebase credentials from .env
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "venv/serviceAccountKey.json")

# Debugging output
print(f"FIREBASE_DB_URL: {FIREBASE_DB_URL}")  # Should print the URL
print(f"SERVICE_ACCOUNT_PATH: {SERVICE_ACCOUNT_PATH}")  # Should show correct path

# Validate environment variables
if not FIREBASE_DB_URL:
    raise ValueError("❌ ERROR: FIREBASE_DB_URL is missing from .env!")

if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise ValueError(f"❌ ERROR: Service account key not found at {SERVICE_ACCOUNT_PATH}")

# Initialize Firebase
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
print("✅ Firebase Initialized Successfully!")

# Reference the database and write test data
ref = db.reference("/test")
ref.set({"status": "connected"})

print("✅ Data successfully sent to Firebase!")
