import firebase_admin
from firebase_admin import credentials, db

# Load Firebase credentials (Make sure you have the JSON file in your directory)
cred = credentials.Certificate("/home/Agrisense/Thesis/venv/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://agrisense-6a089-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Reference the database and write test data
ref = db.reference('/test')
ref.set({
    'status': 'connected'
})

print("Data successfully sent to Firebase!")

