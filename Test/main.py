import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types # For creating message Content/Parts
from dotenv import load_dotenv

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

load_dotenv()



async def main():
    # Step 1: Define Tools for Sub-Agents

def get_weather(city: str):
    print(f"---Tool: get_weather call for city: {city}---")
    city_normalized = city.lower().replace(" ", "")
    mock_weather = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather:
        return mock_weather[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

    def say_hello(name: str = "there"):
        print(f"---Tool: say_hello call for name: {name}---")
        return f"Hello, {name}!"

    def say_goodbye():
        print(f"---Tool: say_goodbye call---")
        return f"Goodbye, Have a great day!"


    def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
        """Retrieves weather, converts temp unit based on session state."""
        print(f"--- Tool: get_weather_stateful called for {city} ---")

        # --- Read preference from state ---
        preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius") # Default to Celsius
        print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

        city_normalized = city.lower().replace(" ", "")

        # Mock weather data (always stored in Celsius internally)
        mock_weather_db = {
            "newyork": {"temp_c": 25, "condition": "sunny"},
            "london": {"temp_c": 15, "condition": "cloudy"},
            "tokyo": {"temp_c": 18, "condition": "light rain"},
        }

        if city_normalized in mock_weather_db:
            data = mock_weather_db[city_normalized]
            temp_c = data["temp_c"]
            condition = data["condition"]

            # Format temperature based on state preference
            if preferred_unit == "Fahrenheit":
                temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
                temp_unit = "°F"
            else: # Default to Celsius
                temp_value = temp_c
                temp_unit = "°C"

            report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
            result = {"status": "success", "report": report}
            print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

            # Example of writing back to state (optional for this tool)
            tool_context.state["last_city_checked_stateful"] = city
            print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

            return result
        else:
            # Handle city not found
            error_msg = f"Sorry, I don't have weather information for '{city}'."
            print(f"--- Tool: City '{city}' not found. ---")
            return {"status": "error", "error_message": error_msg}

    print("✅ State-aware 'get_weather_stateful' tool defined.")

    # Step 2: Define the Sub-Agents (Greeting & Farewell)
    AGENT_MODEL = os.getenv("MODEL_GPT_4_O_MINI")
    greeting_agent = None
    try:
        greeting_agent = Agent(
            model=os.getenv("MODEL_GEMINI_2_0_FLASH"),
            name="greeting_agent",
            instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting using the 'say_hello' tool. Do nothing else.",
            description="Handles simple greetings and hellos using the 'say_hello' tool.",
            tools=[say_hello],
        )
        print(f"✅ Agent '{greeting_agent.name}' redefined.")
    except Exception as e:
        print(f"❌ Could not redefine Greeting agent. Error: {e}")

    # --- Redefine Farewell Agent (from Step 3) ---
    farewell_agent = None
    try:
        farewell_agent = Agent(
            model=os.getenv("MODEL_GEMINI_2_0_FLASH"),
            name="farewell_agent",
            instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message using the 'say_goodbye' tool. Do not perform any other actions.",
            description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
            tools=[say_goodbye],
        )
        print(f"✅ Agent '{farewell_agent.name}' redefined.")
    except Exception as e:
        print(f"❌ Could not redefine Farewell agent. Error: {e}")

    # Step 3: Define the Root Agent with Sub-Agents
    if greeting_agent and farewell_agent:
        root_agent_model = os.getenv("MODEL_GEMINI_2_0_FLASH")
        root_agent_stateful = Agent(
            name="weather_agent_v4_stateful",
            model = root_agent_model,
            description="Main agent: Provides weather (state-aware unit), delegates greetings/farewells, saves report to state.",
            instruction="You are the main Weather Agent. Your job is to provide weather using 'get_weather_stateful'. "
                        "The tool will format the temperature based on user preference stored in state. "
                        "Delegate simple greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
                        "Handle only weather requests, greetings, and farewells.",
            tools=[get_weather_stateful], # Use the state-aware tool
            sub_agents=[greeting_agent, farewell_agent], # Include sub-agents
            output_key="last_weather_report"
        )
        print(f"✅ Root agent '{root_agent_stateful.name}' created using model '{root_agent_model}'.")
    else:
        print("❌ Cannot create stateful root agent. Prerequisites missing.")
        if not greeting_agent: print(" - greeting_agent definition missing.")
        if not farewell_agent: print(" - farewell_agent definition missing.")
        if 'get_weather_stateful' not in globals(): print(" - get_weather_stateful tool missing.")
    # Step 4: Interact with the Agent Team
    session_service_stateful = InMemorySessionService()
    print("✅ New InMemorySessionService created for state demonstration.")

    APP_NAME = "weather_tutorial_agent_team"

    SESSION_ID_STATEFUL = "session_state_demo_001"
    USER_ID_STATEFUL = "user_state_demo"  

    initial_state = {
        "user_preference_temperature_unit": "Celsius"
    }  

    session_stateful = session_service_stateful.create_session(
        app_name=APP_NAME, # Use the consistent app name
        user_id=USER_ID_STATEFUL,
        session_id=SESSION_ID_STATEFUL,
        state=initial_state # <<< Initialize state during creation
    )
    print(f"✅ Session '{SESSION_ID_STATEFUL}' created for user '{USER_ID_STATEFUL}'.")
    retrieved_session = session_service_stateful.get_session(app_name=APP_NAME,
                                                        user_id=USER_ID_STATEFUL,
                                                        session_id = SESSION_ID_STATEFUL)
    print("\n--- Initial Session State ---")
    if retrieved_session:
        print(retrieved_session.state)
    else:
        print("Error: Could not retrieve session.")

    runner_root_stateful = Runner(
        agent=root_agent_stateful,
        app_name=APP_NAME,
        session_service=session_service_stateful # Use the NEW stateful session service
    )
    print(f"✅ Runner created for stateful root agent '{runner_root_stateful.agent.name}' using stateful session service.")


    async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
        """Sends a query to the agent and prints the final response."""
        print(f"\n>>> User Query: {query}")

        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."

        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break

        print(f"<<< Agent Response: {final_response_text}")

    async def run_stateful_conversation():
        print("\n--- Testing State: Temp Unit Conversion & output_key ---")

        # 1. Check weather (Uses initial state: Celsius)
        print("--- Turn 1: Requesting weather in London (expect Celsius) ---")
        await call_agent_async("What's the weather in London?", 
                             runner_root_stateful,
                             USER_ID_STATEFUL,
                             SESSION_ID_STATEFUL)

        # 2. Manually update state preference to Fahrenheit - DIRECTLY MODIFY STORAGE
        print("\n--- Manually Updating State: Setting unit to Fahrenheit ---")
        try:
            # Access the internal storage directly - THIS IS SPECIFIC TO InMemorySessionService for testing
            stored_session = session_service_stateful.sessions[APP_NAME][USER_ID_STATEFUL][SESSION_ID_STATEFUL]
            stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"
            print(f"--- Stored session state updated. Current 'user_preference_temperature_unit': {stored_session.state['user_preference_temperature_unit']} ---")
        except KeyError:
            print(f"--- Error: Could not retrieve session '{SESSION_ID_STATEFUL}' from internal storage for user '{USER_ID_STATEFUL}' in app '{APP_NAME}' to update state. Check IDs and if session was created. ---")
        except Exception as e:
            print(f"--- Error updating internal session state: {e} ---")

        # 3. Check weather again (Tool should now use Fahrenheit)
        print("\n--- Turn 2: Requesting weather in New York (expect Fahrenheit) ---")
        await call_agent_async("Tell me the weather in New York.",
                             runner_root_stateful,
                             USER_ID_STATEFUL,
                             SESSION_ID_STATEFUL)

        # 4. Test basic delegation (should still work)
        print("\n--- Turn 3: Sending a greeting ---")
        await call_agent_async("Hi!",
                             runner_root_stateful,
                             USER_ID_STATEFUL,
                             SESSION_ID_STATEFUL)

  # Execute the conversation
    await run_stateful_conversation()

    # Inspect final session state after the conversation
    print("\n--- Inspecting Final Session State ---")
    final_session = session_service_stateful.get_session(app_name=APP_NAME,
                                                        user_id= USER_ID_STATEFUL,
                                                        session_id=SESSION_ID_STATEFUL)
    if final_session:
        print(f"Final Preference: {final_session.state.get('user_preference_temperature_unit')}")
        print(f"Final Last Weather Report (from output_key): {final_session.state.get('last_weather_report')}")
        print(f"Final Last City Checked (by tool): {final_session.state.get('last_city_checked_stateful')}")
        # Print full state for detailed view
        # print(f"Full State: {final_session.state}")
    else:
        print("\n❌ Error: Could not retrieve final session state.")

if __name__ == "__main__":
    asyncio.run(main())
