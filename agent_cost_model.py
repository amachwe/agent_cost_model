from matplotlib import pyplot as plt
from numpy import random

## Constants ##
class CostModel:
    MODEL_NAME: str
    INPUT_TOKEN_COST: float
    OUTPUT_TOKEN_COST: float
    PER_TOKEN: int
    ACTUAL_INPUT_TOKEN_COST: float
    ACTUAL_OUTPUT_TOKEN_COST: float

class Gemini25FlashCost(CostModel):
    MODEL_NAME = "gemini-2.5-flash"
    INPUT_TOKEN_COST = 0.30
    OUTPUT_TOKEN_COST = 2.50
    PER_TOKEN = 1_000_000
    ACTUAL_INPUT_TOKEN_COST = INPUT_TOKEN_COST / PER_TOKEN
    ACTUAL_OUTPUT_TOKEN_COST = OUTPUT_TOKEN_COST / PER_TOKEN

class Gemini25FlashUltraCost(CostModel):
    MODEL_NAME = "gemini-2.5-flash-lite"
    INPUT_TOKEN_COST = 0.10
    OUTPUT_TOKEN_COST = 0.40
    PER_TOKEN = 1_000_000
    ACTUAL_INPUT_TOKEN_COST = INPUT_TOKEN_COST / PER_TOKEN
    ACTUAL_OUTPUT_TOKEN_COST = OUTPUT_TOKEN_COST / PER_TOKEN


def single_turn_model(total_input_tokens_base: int, total_input_tokens_data: int,
          total_output_tokens_base: int, model:CostModel) -> float:
    """
    Calculate the cost of using the model based on token usage.
    """

    base_input = total_input_tokens_base* model.ACTUAL_INPUT_TOKEN_COST
    input_cost = total_input_tokens_data * model.ACTUAL_INPUT_TOKEN_COST
    output_cost = total_output_tokens_base * model.ACTUAL_OUTPUT_TOKEN_COST

    total_cost = base_input + input_cost + output_cost
    return total_cost, base_input, input_cost, output_cost

def single_turn_model_with_guardrails(total_input_tokens_base: int, total_input_tokens_data: int,
          total_output_tokens_base: int, model:CostModel, guardrail_model: CostModel, multiplier=1.0) -> float:
    """
    Calculate the cost of using the model based on token usage including guardrail costs.
    Guardrails - are assumed to process user input + data and output + input data for context.
    Multiplier - to account for multiple guardrail checks per turn.
    """
    base_input = total_input_tokens_base* model.ACTUAL_INPUT_TOKEN_COST
    input_cost = total_input_tokens_data * model.ACTUAL_INPUT_TOKEN_COST
    output_cost = total_output_tokens_base * model.ACTUAL_OUTPUT_TOKEN_COST
    input_guardrail_cost = (total_input_tokens_data + total_input_tokens_base) * guardrail_model.ACTUAL_INPUT_TOKEN_COST*multiplier
    output_guardrail_cost = total_output_tokens_base * guardrail_model.ACTUAL_OUTPUT_TOKEN_COST*multiplier

    total_cost = base_input + input_cost + output_cost + input_guardrail_cost + output_guardrail_cost
    return total_cost, base_input, input_cost, output_cost, input_guardrail_cost, output_guardrail_cost

def single_turn_agent(number_of_turns:int, single_turn_model_cost: float) -> float:
    """
    Calculate the cost of using an agent based on number of turns and single turn model cost.
    """
    return number_of_turns * single_turn_model_cost

def single_turn_agent_increment(number_of_turns:int, base_input_tokens: int, data_input_tokens: int, output_tokens: int, model: CostModel, with_guardrails=False, guardrail_model=None, multiplier=1.0) -> float:
    """
    Calculate the cost of using an agent based on number of turns and single turn model cost.
    """
    total_cost = 0
    for turn in range(1, number_of_turns+1):
        if with_guardrails and guardrail_model is not None:
            turn_cost = single_turn_model_with_guardrails(base_input_tokens, turn*data_input_tokens, output_tokens, model, guardrail_model,multiplier)
        else:
            turn_cost = single_turn_model(base_input_tokens, turn*data_input_tokens, output_tokens, model)
        total_cost += turn_cost[0]
    
    return total_cost

def multi_turn_agent(number_of_interactions:int, single_turn_agent_cost: float) -> float:
    """
    Calculate the cost of using a multi-turn agent based on number of interactions and single turn agent cost.
    """
    return number_of_interactions * single_turn_agent_cost




if __name__ == "__main__":
    import tqdm


    cost_model  = Gemini25FlashCost()
    guardrail_model = Gemini25FlashUltraCost()


    # Configuration variables
    SAMPLE = 1_000_000
    GUARDRAIL_MULTIPLIER = 2.0 # to account for multiple guardrail checks per turn - assuming 2 for this analysis
    MEAN_DATA_INPUT_TOKENS = 200
    STD_DATA_INPUT_TOKENS = 20
    MEAN_BASE_OUTPUT_TOKENS = 150
    STD_BASE_OUTPUT_TOKENS = 20
    N_TURNS_POISSON_LAMBDA = 3
    N_AGENT_INTERACTIONS_POISSON_LAMBDA = 2
    WITH_GUARDRAILS = True
    BASE_INPUT_TOKEN_COUNT_CONSTANTS = [100,500, 1000, 1500, 2000, 3000]
    

    # Storage for costs
    costs = []

    # Generate random samples
    base_input_tokens = random.choice(BASE_INPUT_TOKEN_COUNT_CONSTANTS, size=SAMPLE) # Base input tokens = fixed prompt template a set of possible sizes based on input source
    
    data_input_tokens = abs(random.normal(loc=MEAN_DATA_INPUT_TOKENS, scale=STD_DATA_INPUT_TOKENS, size=SAMPLE).astype(int)) # Input tokens from RAG, user, data, etc. variable quantity per turn
    base_output_tokens = abs(random.normal(loc=MEAN_BASE_OUTPUT_TOKENS, scale=STD_BASE_OUTPUT_TOKENS, size=SAMPLE).astype(int)) # Output tokens from model per turn variable quantity per turn
    
    # n_turns = random.choice([1,2,3,4], size=SAMPLE) # uniform random choice with at least 1 turn
    n_turns = random.poisson(lam=N_TURNS_POISSON_LAMBDA, size=SAMPLE).astype(int) + 1 # at least 1 turn
    
    # n_agent_interactions = random.choice([2,3,4,5,6], size=SAMPLE) # uniform random choice with at least 2 interactions
    n_agent_interactions = random.poisson(lam=N_AGENT_INTERACTIONS_POISSON_LAMBDA, size=SAMPLE).astype(int) + 1 # at least 1 interaction
 
    for i in tqdm.tqdm(range(SAMPLE)):
        if base_output_tokens[i] < 0 or data_input_tokens[i] < 0:
            print(f"Base Input Tokens: {base_input_tokens}")
            print(f"Data Input Tokens: {data_input_tokens}")
            print(f"Base Output Tokens: {base_output_tokens}")
        single_turn_agent_cost = single_turn_agent_increment(n_turns[i], base_input_tokens[i], data_input_tokens[i], base_output_tokens[i], cost_model, with_guardrails=WITH_GUARDRAILS, guardrail_model=guardrail_model, multiplier=GUARDRAIL_MULTIPLIER)
        multi_turn_agent_cost = multi_turn_agent(n_agent_interactions[i], single_turn_agent_cost)

        costs.append(multi_turn_agent_cost)

    print(f"Average Cost per Multi-Turn Agent Interaction: ${sum(costs)/len(costs):.7f}")
    print(f"Median Cost per Multi-Turn Agent Interaction: ${sorted(costs)[len(costs)//2]:.7f}")
    print(f"Max Cost per Multi-Turn Agent Interaction: ${max(costs):.7f}")
    print(f"Min Cost per Multi-Turn Agent Interaction: ${min(costs):.7f}")
    print(f"Total Cost for {SAMPLE} Multi-Turn Agent Interactions: ${sum(costs):.7f}") 

    fig, axs = plt.subplots(2,3, sharey=True)
    axs[0][0].hist(base_input_tokens, bins=50, alpha=0.7)
    axs[0][0].set_title("Distribution of Base Input Tokens")
    axs[0][0].set_xlabel("Number of Tokens")
 

    axs[0][1].hist(data_input_tokens, bins=50, alpha=0.7)
    axs[0][1].set_title("Distribution of Data Input Tokens")        
    axs[0][1].set_xlabel("Number of Tokens")
  

    axs[0][2].hist(base_output_tokens, bins=50, alpha=0.7)
    axs[0][2].set_title("Distribution of Base Output Tokens")           
    axs[0][2].set_xlabel("Number of Tokens")   
                

    axs[1][0].hist(n_turns, bins=50, alpha=0.7, color='green')
    axs[1][0].set_title("Distribution of Number of Turns")      
    axs[1][0].set_xlabel("Number of Turns") 
 

    axs[1][1].hist(n_agent_interactions, bins=50, alpha=0.7, color='green')
    axs[1][1].set_title("Distribution of Number of Agent Interactions") 
    axs[1][1].set_xlabel("Number of Interactions")  


    axs[1][2].hist(costs, bins=50, alpha=0.7, color='orange')
    axs[1][2].set_title("Distribution of Multi Turn Agent Costs")
    axs[1][2].set_xlabel("Cost ($)")
    axs[1][2].set_ylabel("Frequency")

    plt.title(f"Agent Cost Analysis {cost_model.MODEL_NAME}")
    plt.show()

    
    

    

    


