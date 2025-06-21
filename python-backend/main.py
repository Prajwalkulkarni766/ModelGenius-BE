from fastapi import FastAPI
import ollama

app = FastAPI()

@app.get("/")
async def root():
    return { "message" : "Hello world" }

@app.post("/generate")
async def generate(prompt: str):
    response = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": prompt
    }])
    # return { "response": response["message"]["content"]}
    return response