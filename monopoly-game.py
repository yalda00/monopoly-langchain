# monopoly simulator
from simulator.monosim.player import Player
from simulator.monosim.board import get_board, get_roads, get_properties, get_community_chest_cards, get_bank

# LangChain
from langchain_openai import ChatOpenAI
from langchain import OpenAI, LLMChain, PromptTemplate

# other imports
from langchain_community.llms import OpenAI as ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# dotenv for loading API keys
import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Define the API keys
openai_key = os.getenv("OPENAI_API_KEY")

# Fast and slow minds
def fast_mind(prompt: str, model="gpt-4o-mini") -> dict:
    """Fast mind for quick decisions."""
    model_instance = ChatOpenAI(model=model, api_key=openai_key)
    response = model_instance.predict(prompt)
    uncertainty = float(response.metadata.get("uncertainty", 0.0))
    decision = response.text.strip()
    return {"decision": decision, "uncertainty": uncertainty}

def slow_mind(prompt: str, model="gpt-4") -> dict:
    """Slow mind for complex decisions."""
    model_instance = ChatOpenAI(model=model, api_key=openai_key)
    response = model_instance.predict(prompt)
    decision = response.text.strip()
    return {"decision": decision}

def custom_buy(self, dict_property_info: dict) -> str | None:
    player_state = self.get_state()
    cash = player_state['cash']
    price = dict_property_info['price']

    if dict_property_info['belongs_to'] is not None:
        return None

    if cash < price:
        return None

    if dict_property_info['type'] == 'road':
        color = dict_property_info['color']
        owned_roads = player_state['owned_roads']

        owned_same_color = sum(
            1 for road in owned_roads
            if self._dict_properties.get(road, {}).get('color') == color
        )

        total_color_properties = sum(
            1 for prop in self._dict_properties.values()
            if prop.get('color') == color
        )

        if owned_same_color + 1 == total_color_properties:
            return "buy"

    if dict_property_info['type'] in ['station', 'utility']:
        return "buy"

    return None

    
    # currently, no trading. So don't buy property if someone else owns it
    if dict_property_info['belongs_to'] != None:
        return None

    # If the property isn't owned, I get the LLM to decide whether to buy it
    # Steps: 1. making a prompt template, 2. injecting data, 3. setting up 
    # output parsing, and 4. running an LLM.

    ##################################################
    # Step 1: Making a prompt template
    ##################################################

    # I'm trying to keep this prompt template general to reuse it for other games
    template = """
    You are a strategic decision-maker playing a game of {game}.

    Rules of the game:
    {rules} 
    
    Your goal:
    Maximise your chances of winning {game} with forward-thinking reasoning.

    Your task:
    Analyse whether to buy {property_name} given the data that follows.

    Property Information:
    - Cost: {price}
    - Base rent: {rent}
    
    Provide an uncertainty score reflecting your confidence in the decision. The score should be accurate and align with your analysis.
    
    Reason whether you should buy {property_name}. Then output yes/no as your final answer.
    """

    # Only add color data when available (for roads)
    if (dict_property_info['type'] == 'road'):
        template += "\n- Color: {color}\n- Number of {color} properties you own: {n_color_properties}\n\nYou currently have {cash} in cash and {n_roads} roads."
    elif (dict_property_info['type'] == 'station'):
        template += "\n\nYou currently have {cash} in cash and {n_stations} stations."
    elif (dict_property_info['type'] == 'utility'):
        template += "\n\nYou currently have {cash} in cash and {n_utilities} utilities."
    template += "\nReason whether you should buy {property_name}. Then output yes/no as your final answer."

    
    ##################################################
    # Step 2: Injecting data
    ##################################################

    inject = {
        'agent_name': self.get_state()['name'],
        'game': 'Monopoly',
        # Insert better rules later
        'rules': 'Typical Monopoly rules, but you cannot trade with other players or mortgage properties.',
        'property_name': dict_property_info['name'],
        'price': dict_property_info['price'],
        'rent': dict_property_info['rent'],
        'cash': self.get_state()['cash'],
        'n_roads': len(self.get_state()['owned_roads']),
        'n_stations': len(self.get_state()['owned_stations']),
        'n_utilities': len(self.get_state()['owned_utilities'])
    }

    # Only add color data when available (for roads)
    if (dict_property_info['type'] == 'road'):
        inject['color'] = dict_property_info['color']
        
        # Bugs here! self._dict_properties.get(property) fails for some
        # properties. Don't reuse this code!
        n_color_properties = 0
        for property in self.get_state()['owned_roads']:
            prop_info = self._dict_properties.get(property)
            if prop_info and prop_info['color'] == dict_property_info['color']:
                n_color_properties += 1
        inject['n_color_properties'] = n_color_properties

    prompt_template = PromptTemplate(input_variables=inject.keys(), template=template)
    prompt = prompt_template.format(**inject)

    
    ##################################################
    # Step 3: Setting up output parsing
    ##################################################

    class Output(BaseModel):
        reasoning: str = Field(description="Your reasoning for the decision")
        decision: str = Field(description="Whether to buy the property (yes/no)")
    
    ##################################################
    # Step 4: Running the LLM
    ##################################################

    # Fast mind decision
    fast_response = fast_mind(prompt)

    # Check uncertainty
    if fast_response['uncertainty'] > 60.0:

        slow_response = slow_mind(prompt)
        decision = slow_response['decision']
    else:
        decision = fast_response['decision']

    return "buy" if decision == "yes" else None

def modify_buy_or_bid(buy) -> str:
    return custom_buy

class CustomPlayer(Player):
    buy_or_bid = modify_buy_or_bid(Player.buy_or_bid)

def initialize_game() -> dict:
    """
    Initializes a game with two players and sets up the bank, board, roads, properties, 
    and community chest cards.
    
    Returns:
        dict: A dictionary containing the following:
            - "bank": Game's bank object.
            - "board": Main game board.
            - "roads": List of road objects.
            - "properties": List of property objects.
            - "community_chest_cards": Dictionary of community chest cards.
            - "players": List of two Player objects, with Player 1 first.
    """
    
    bank = get_bank()
    board = get_board()
    roads = get_roads()
    properties = get_properties()
    community_chest_cards = get_community_chest_cards()
    community_cards_deck = list(community_chest_cards.keys())

    # Note how we have one of our players vs. a default player that just buys 
    # whenever cash is available. We can change this later
    player1 = CustomPlayer('Alice', 1, bank, board, roads, properties, community_cards_deck)
    player2 = Player('Bob', 2, bank, board, roads, properties, community_cards_deck)
    
    player1.meet_other_players([player2])
    player2.meet_other_players([player1])
    
    return {
        "bank": bank,
        "board": board,
        "roads": roads,
        "properties": properties,
        "community_chest_cards": community_chest_cards,
        "players": [player1, player2]
    }

def get_current_state(players) -> dict:
    """
    Retrieves the current state of each player, including position, owned roads, 
    money, mortgaged properties, and other status details.

    Args:
        players (list[Player]): List of Player objects in the game.

    Returns:
        dict: A dictionary containing:
            - "players": A list of dictionaries, each with a player's state.
    """
    
    current_state = {
        "players": [{"state": player.get_state()} for player in players]
    }
    return current_state

game = initialize_game()

player1 = game["players"][0]
player2 = game["players"][1]
list_players = [player1, player2]

# WARNING: KEEP THIS UNDER 10 ROUNDS FOR NOW
# EACH ROUND COSTS US MONEY
STOP_AT_ROUND = 2

idx_count = 0
while not player1.has_lost() and not player2.has_lost() and idx_count < STOP_AT_ROUND:
    for player in list_players:
        # Uncomment this line to run the game if you're sure there are no bugs
        player.play() 
    idx_count += 1