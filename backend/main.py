import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from debugger import PythonDebugger
import numpy as np
import pickle

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] if you're using Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for frontend (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the pre-trained model
MODEL_PATH = "models/debugger_model.pkl"  # Path to your trained model
with open(MODEL_PATH, "rb") as f:
    error_type_model = pickle.load(f)

# Debugging: Confirm the loaded model type
print("Loaded model type:", type(error_type_model))

# Serve the frontend at the root route
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FastAPI Debugger</title>
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>
        <h1>Welcome to the AI Python Debugger</h1>
        <form id="debug-form">
            <label for="code">Enter Python Code:</label><br>
            <textarea id="code" name="code" rows="10" cols="50"></textarea><br>
            <button type="button" onclick="submitCode()">Debug</button>
        </form>
        <div id="debug-output"></div>
        <script>
            async function submitCode() {
                const code = document.getElementById("code").value;
                try {
                    const response = await fetch('/debug', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ code }),
                    });
                    const result = await response.json();
                    document.getElementById("debug-output").innerText = JSON.stringify(result, null, 2);
                } catch (error) {
                    document.getElementById("debug-output").innerText = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

class DebugRequest(BaseModel):
    code: str

@app.post("/debug")
async def debug_code(request: DebugRequest):
    code = request.code
    debugger = PythonDebugger(code)
    
    # Step 1: Check for syntax errors
    syntax_error = debugger.find_syntax_errors()
    response = {"syntax_error": syntax_error}
    
    # Step 2: Use the pre-trained model to predict error type
    if syntax_error:
        feature_vector = np.array([[
            len(code),  # original_code_length
            len(code),  # changed_code_length
            0,          # code_length_difference
            0           # change_count
        ]])
        predicted_error_type = error_type_model.predict(feature_vector)[0]
        response["predicted_error_type"] = predicted_error_type

    # Step 3: Get debugging suggestions from OpenAI
    openai_prompt = f"Provide debugging suggestions for the following code and error message:\n\nCode:\n{code}\n\nError:\n{syntax_error}"
    openai_response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for debugging Python code."},
            {"role": "user", "content": openai_prompt}
        ],
        max_tokens=200
    )
    response["openai_suggestions"] = openai_response.choices[0].message.content.strip()
    
    return response

# Automatically run the Uvicorn app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
