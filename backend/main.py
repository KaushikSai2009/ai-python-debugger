import os
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from debugger import PythonDebugger
import pickle
import numpy as np

app = FastAPI()

# Load the pre-trained ML model
MODEL_PATH = "error_type_model.pkl"  # Path to your trained model
with open(MODEL_PATH, "rb") as f:
    error_type_model = pickle.load(f)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class DebugRequest(BaseModel):
    code: str

@app.post("/debug")
async def debug_code(request: DebugRequest):
    code = request.code
    debugger = PythonDebugger(code)
    
    # Step 1: Check for syntax errors
    syntax_error = debugger.find_syntax_errors()
    response = {"syntax_error": syntax_error}
    
    # Step 2: Use AI Model to predict error type
    if syntax_error:
        # Preprocess the error message to a feature vector (example logic)
        error_vector = np.array([len(syntax_error), syntax_error.count(" ")])  # Example: length and space count
        error_type = error_type_model.predict([error_vector])[0]
        response["predicted_error_type"] = error_type
    
    # Step 3: Get suggestions from OpenAI
    openai_prompt = f"Provide debugging suggestions for the following code and error message:\n\nCode:\n{code}\n\nError:\n{syntax_error}"
    openai_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        max_tokens=200
    )
    response["openai_suggestions"] = openai_response["choices"][0]["text"].strip()
    
    return response

# Automatically run the Uvicorn app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)