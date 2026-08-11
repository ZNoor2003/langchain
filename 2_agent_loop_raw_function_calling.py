
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


from openai import OpenAI
from langfuse import observe
import json

client = OpenAI() 

MAX_ITERATIONS = 10
MODEL = "gpt-5.4-mini"

# --- Tools (Langchain @tool decorator) ---

@observe(as_type="tool") 
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >>Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard":89.50}
    return prices.get(product, 0.0) # Return 0.0 if product not found

@observe(as_type="tool") 
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the discounted price.
    Available tiers: bronze, silver, gold."""
    print(f"   >>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2) # Return discounted price rounded to 2 decimal places

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what Langchain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. laptop, headphones, keyboard"
                    }
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the discounted price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "discount_tier": {
                        "type": "string",
                        "enum": ["bronze", "silver", "gold"]
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]


# NOTE: OpenAI can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# So OpenAI can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           products: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """

# --- Helper: traced OpenAI call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangFuse.

@observe(name="OpenAI Chat", as_type="generation")
def openai_chat_traced(messages):
    response = client.chat.completions.create(
        model=MODEL,
        tools=tools_for_llm,
        messages=messages
    )
    return response.choices[0].message


# --- Agent Loop ---

@observe(name="LangChain Agent Loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "Available products in the catalog: laptop, headphones, keyboard. "
                "When a user mentions any of these products (even generically, "
                "like 'a laptop'), use that exact product name to call the tool.\n\n"
                "STRICT RULES - you must use these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call the get_product_price tool to look up the real price of the product.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool to calculate the discounted price.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            ),
        },
        {"role":"user", "content": question},
    ]


    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: openai.chat() directly instead of llm_with_tools.invoke()
        ai_message = openai_chat_traced(messages=messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f" [Tool Result] {observation}")

        messages.append(ai_message.model_dump())
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": str(observation),
            }
        )

    print("ERROR: Maximum iterations reached without a final answer.")
    return None
    

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")