import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, validator
from fastapi import FastAPI, HTTPException
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("Missing OpenAI API key")
    raise EnvironmentError("Ensure OPENAI_API_KEY is set.")

# FastAPI app
app = FastAPI()

# Pydantic models for validation
class Query(BaseModel):
    question: str

    @validator("question")
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v

class Query_Response(BaseModel):
    result: str

def process_query(question: str) -> str:
    try:
        logger.info(f"Processing question: {question}")

        # Connect to the database
        db = SQLDatabase.from_uri("sqlite:///stocksDB.db")
        logger.info(f"Connected to database. Tables: {db.get_usable_table_names()}")

        # Create OpenAI LLM
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Create the agent using the OpenAI LLM
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            verbose=True,
            handle_parsing_errors=True
        )

        # Run the query
        final_state = agent_executor.invoke(question)
        return final_state.get("output", "No output generated.")
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return "An error occurred while processing your request."

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/question", response_model=Query_Response)
async def question_api(q: Query):
    try:
        result = process_query(q.question)
        return Query_Response(result=result)
    except Exception as e:
        logger.error(f"Error processing API request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

def interactive_mode():
    """Run the app in interactive mode."""
    print("Welcome to the OpenAI SQL Query Processor!")
    print("Type your question below or 'exit' to quit.")

    while True:
        # Get user input
        question = input("\nYour question: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Exiting. Goodbye!")
            break

        # Validate and process the question
        try:
            validated_query = Query(question=question)
            response = process_query(validated_query.question)
            print(f"\nResponse: {response}")
        except ValueError as e:
            print(f"Invalid input: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # Run in interactive mode
        interactive_mode()
    else:
        # Run FastAPI with uvicorn
        import uvicorn
        uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")