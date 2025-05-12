from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List
from fastapi.responses import HTMLResponse

# Load the model and scaler
model = joblib.load('credit_score_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_credit_score(
    Age: int = Form(...),
    Annual_Income: float = Form(...),
    Monthly_Inhand_Salary: float = Form(...),
    Num_Bank_Accounts: int = Form(...),
    Num_Credit_Card: int = Form(...),
    Interest_Rate: float = Form(...),
    Num_of_Loan: int = Form(...),
    Delay_from_due_date: int = Form(...),
    Num_of_Delayed_Payment: int = Form(...),
    Changed_Credit_Limit: int = Form(...),
    Num_Credit_Inquiries: int = Form(...),
    Credit_Mix: int = Form(...),
    Outstanding_Debt: float = Form(...),
    Credit_Utilization_Ratio: float = Form(...),
    Credit_History_Age: int = Form(...),
    Payment_of_Min_Amount: int = Form(...),
    Total_EMI_per_month: float = Form(...),
    Amount_invested_monthly: float = Form(...),
    Payment_Behaviour: int = Form(...),
    Monthly_Balance: float = Form(...),
    Month: int = Form(...),
    Occupation: int = Form(...),
    Type_of_Loan: int = Form(...),
    Num_Credit_Inquiries_Last_6_Months: int = Form(...),
    Avg_Credit_Card_Utilization_Last_Year: float = Form(...),
    Num_Late_Payments_Last_12_Months: int = Form(...),
    Has_Active_Loan: int = Form(...)
):
    # Create input array
    input_array = np.array([[
        Age, Annual_Income, Monthly_Inhand_Salary,
        Num_Bank_Accounts, Num_Credit_Card, Interest_Rate,
        Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment,
        Changed_Credit_Limit, Num_Credit_Inquiries, Credit_Mix,
        Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age,
        Payment_of_Min_Amount, Total_EMI_per_month, Amount_invested_monthly,
        Payment_Behaviour, Monthly_Balance, Month, Occupation, Type_of_Loan,
        Num_Credit_Inquiries_Last_6_Months, Avg_Credit_Card_Utilization_Last_Year,
        Num_Late_Payments_Last_12_Months, Has_Active_Loan
    ]])

    # Scale the input data
    scaled_input = scaler.transform(input_array)

    # Make a prediction
    prediction = model.predict(scaled_input)

    # Map the prediction to a credit score category
    credit_score_mapping = {0: "Poor", 1: "Standard", 2: "Good"}
    predicted_category = credit_score_mapping.get(prediction[0], "Unknown")

    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Score Prediction Result</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f3f3f3; text-align: center; padding: 50px; }}
            .result-container {{ background: #ffffff; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2); }}
            .score-result {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        </style>
    </head>
    <body>
        <div class="result-container">
            <h2>Prediction Result</h2>
            <p class="score-result">Predicted Credit Score: {predicted_category}</p>
            <a href="/">Predict Again</a>
        </div>
    </body>
    </html>
    """)

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Score Prediction</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background-image: url('https://media.istockphoto.com/id/531236924/photo/group-of-credit-cards-on-computer-keyboard.jpg?s=612x612&w=0&k=20&c=5iAuEH7ipVgVDI9TkgzTC8Xx0roMhvDlT79UzRiSzcE='); 
                background-size: cover; 
                background-attachment: fixed; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                min-height: 100vh; 
                margin: 0; 
            }
            form { 
                max-width: 600px; 
                background: rgb(235, 228, 228); 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2); 
            }
            label { display: block; margin-top: 10px; }
            input { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #4CAF50; color: white; padding: 10px; border: none; margin-top: 15px; cursor: pointer; border-radius: 5px; }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <form action="/predict" method="post">
            <h2 style="text-align:center;">Credit Score Prediction</h2>
            <label>Age:</label>
            <input type="number" name="Age" required>

            <label>Annual Income:</label>
            <input type="number" name="Annual_Income" required>

            <label>Monthly In-hand Salary:</label>
            <input type="number" name="Monthly_Inhand_Salary" required>

            <label>Number of Bank Accounts:</label>
            <input type="number" name="Num_Bank_Accounts" required>

            <label>Number of Credit Cards:</label>
            <input type="number" name="Num_Credit_Card" required>

            <label>Interest Rate:</label>
            <input type="number" name="Interest_Rate" required>

            <label>Number of Loans:</label>
            <input type="number" name="Num_of_Loan" required>

            <label>Delay from Due Date:</label>
            <input type="number" name="Delay_from_due_date" required>

            <label>Number of Delayed Payments:</label>
            <input type="number" name="Num_of_Delayed_Payment" required>

            <label>Changed Credit Limit:</label>
            <input type="number" name="Changed_Credit_Limit" required>

            <label>Number of Credit Inquiries:</label>
            <input type="number" name="Num_Credit_Inquiries" required>

            <label>Credit Mix:</label>
            <input type="number" name="Credit_Mix" required>

            <label>Outstanding Debt:</label>
            <input type="number" name="Outstanding_Debt" required>

            <label>Credit Utilization Ratio:</label>
            <input type="number" name="Credit_Utilization_Ratio" required>

            <label>Credit History Age:</label>
            <input type="number" name="Credit_History_Age" required>

            <label>Payment of Min Amount:</label>
            <input type="number" name="Payment_of_Min_Amount" required>

            <label>Total EMI per Month:</label>
            <input type="number" name="Total_EMI_per_month" required>

            <label>Amount Invested Monthly:</label>
            <input type="number" name="Amount_invested_monthly" required>

            <label>Payment Behaviour:</label>
            <input type="number" name="Payment_Behaviour" required>

            <label>Monthly Balance:</label>
            <input type="number" name="Monthly_Balance" required>

            <label>Month:</label>
            <input type="number" name="Month" required>

            <label>Occupation:</label>
            <input type="number" name="Occupation" required>

            <label>Type of Loan:</label>
            <input type="number" name="Type_of_Loan" required>

            <label>Number of Credit Inquiries (Last 6 Months):</label>
            <input type="number" name="Num_Credit_Inquiries_Last_6_Months" required>

            <label>Average Credit Card Utilization (Last Year):</label>
            <input type="number" name="Avg_Credit_Card_Utilization_Last_Year" required>

            <label>Number of Late Payments (Last 12 Months):</label>
            <input type="number" name="Num_Late_Payments_Last_12_Months" required>

            <label>Has Active Loan (1 for Yes, 0 for No):</label>
            <input type="number" name="Has_Active_Loan" required>

            <button type="submit">Predict Credit Score</button>
        </form>
    </body>
    </html>
    """
