from agents import Agent, Task, AgentExecutor, Runner
from dotenv import load_dotenv
import random

load_dotenv()

@function_tool
def random_number():
    return random.randint(1, 100)

find_even_agent = Agent(
    name="Find Even Agent",
    instructions="Your task is to find an even number",
    tools=[random_number]
)

async def main():
    result = await Runner.run(
        find_even_agent,
        input=f"Generate random numbers until you find an even number",
        max_turns = 20
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())










