"""
Market Analyst Buyer Agent - Concordia Implementation
A data-driven negotiator that counters any seller persona with market analysis.
"""

from concordia.agents import entity_agent_with_logging
from concordia.components import agent as agent_components
from concordia.associative_memory import associative_memory
from concordia.language_model import language_model
from concordia.components import entity_component
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]

@dataclass
class NegotiationResponse:
    """Response from negotiation"""
    status: str
    offer: int
    message: str

class OllamaLanguageModel(language_model.LanguageModel):
    """Ollama implementation for Concordia language model"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        
    def sample_text(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text response using Ollama"""
        try:
            import ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": max_tokens}
            )
            return response['response'].strip()
        except ImportError:
            # Fallback for testing without ollama
            return self._mock_response(prompt)
        except Exception:
            return "I need to analyze this market data further."
    
    def sample_choice(self, prompt: str, choices: List[str]) -> str:
        """Choose from options using Ollama"""
        full_prompt = f"{prompt}\nChoices: {choices}\nAnswer:"
        text = self.sample_text(full_prompt, 50)
        for choice in choices:
            if choice.lower() in text.lower():
                return choice
        return choices[0] if choices else ""
    
    def _mock_response(self, prompt: str) -> str:
        """Mock response for testing without Ollama"""
        if "accept" in prompt.lower():
            return "Based on market verification, I accept this offer."
        elif "reject" in prompt.lower():
            return "This doesn't align with current market realities."
        else:
            return "My market analysis suggests this counter-offer represents fair value."

class BuyerPersonalityComponent(entity_component.ContextComponent):
    """Market Analyst personality definition for Concordia"""
    
    def __init__(self):
        super().__init__()
        self.name = "Market Analyst"
        self.traits = ["data-driven", "strategic", "verification-focused", "budget-aware", "adaptable"]
        self.negotiation_style = "Uses market data and competitor analysis to counter any seller tactic. Verifies all claims with factual research."
        self.catchphrases = [
            "My market verification shows...",
            "Let's examine the actual data on...",
            "Competitor benchmarks indicate...",
            "The numbers suggest a fair value would be...",
            "I need to validate those claims against market reality"
        ]
    
    def make_pre_act_value(self) -> str:
        """Return personality context for LLM"""
        return f"""
        You are a Market Analyst buyer. 
        Personality: {', '.join(self.traits)}
        Style: {self.negotiation_style}
        Key phrases: {', '.join(self.catchphrases)}
        Always: Verify claims with data, reference market benchmarks, stay within budget, adapt to seller's style
        Never: Reveal your exact budget, accept claims without verification, get emotional
        """
    
    def get_state(self):
        """Return current state for serialization"""
        return {
            'name': self.name,
            'traits': self.traits,
            'style': self.negotiation_style,
            'catchphrases': self.catchphrases
        }
    
    def set_state(self, state):
        """Restore from saved state"""
        self.name = state['name']
        self.traits = state['traits']
        self.negotiation_style = state['style']
        self.catchphrases = state['catchphrases']

class NegotiationMemoryComponent(entity_component.ContextComponent):
    """Stores and retrieves negotiation history for Concordia"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.offer_history = []
    
    def add_interaction(self, role: str, message: str, offer: Optional[int] = None):
        """Add new interaction to memory"""
        self.conversation_history.append(f"{role}: {message}")
        if offer is not None:
            self.offer_history.append((role, offer))
    
    def make_pre_act_value(self) -> str:
        """Return memory context for LLM"""
        history_str = "\n".join(self.conversation_history[-6:])  # Last 6 exchanges
        return f"Negotiation History:\n{history_str}" if history_str else "No history yet."
    
    def get_state(self):
        """Return current state for serialization"""
        return {
            'conversation': self.conversation_history,
            'offers': self.offer_history
        }
    
    def set_state(self, state):
        """Restore from saved state"""
        self.conversation_history = state['conversation']
        self.offer_history = state['offers']

class SellerObservationComponent(entity_component.ContextComponent):
    """Processes and analyzes seller messages and offers for Concordia"""
    
    def __init__(self):
        super().__init__()
        self.current_message = ""
        self.current_offer = None
        self.seller_style = "unknown"
    
    def update_observation(self, message: str, offer: Optional[int] = None):
        """Update current observation"""
        self.current_message = message
        self.current_offer = offer
        self._analyze_seller_style(message)
    
    def _analyze_seller_style(self, message: str):
        """Analyze seller's negotiation style"""
        message_lower = message.lower()
        if any(word in message_lower for word in ['premium', 'quality', 'best', 'exclusive']):
            self.seller_style = "skillful_enthusiast"
        elif any(word in message_lower for word in ['now', 'quick', 'last chance', 'final']):
            self.seller_style = "aggressive"
        elif any(word in message_lower for word in ['please', 'understand', 'appreciate', 'partner']):
            self.seller_style = "diplomatic"
        elif any(word in message_lower for word in ['data', 'market', 'research', 'analysis']):
            self.seller_style = "analytical"
        else:
            self.seller_style = "neutral"
    
    def make_pre_act_value(self) -> str:
        """Return observation context for LLM"""
        offer_str = f"₹{self.current_offer:,}" if self.current_offer else "No offer"
        return f"Seller's message: '{self.current_message}'. Offer: {offer_str}. Detected style: {self.seller_style}"
    
    def get_state(self):
        """Return current state for serialization"""
        return {
            'message': self.current_message,
            'offer': self.current_offer,
            'style': self.seller_style
        }
    
    def set_state(self, state):
        """Restore from saved state"""
        self.current_message = state['message']
        self.current_offer = state['offer']
        self.seller_style = state['style']

class StrategicDecisionComponent(entity_component.ContextComponent):
    """Implements negotiation strategy and decision logic for Concordia"""
    
    def __init__(self):
        super().__init__()
        self.budget = None
        self.product = None
        self.round_num = 0
    
    def initialize_negotiation(self, product: Product, budget: int):
        """Initialize negotiation parameters"""
        self.product = product
        self.budget = budget
        self.round_num = 0
    
    def increment_round(self):
        """Increment round counter"""
        self.round_num += 1
    
    def make_decision(self, seller_offer: Optional[int], seller_style: str) -> Tuple[str, int]:
        """Make strategic decision based on current state"""
        if seller_offer is None:
            # Opening offer strategy
            base_offer = int(self.product.base_market_price * 0.65)
            return "offer", min(base_offer, self.budget)
        
        if self._should_accept(seller_offer, seller_style):
            return "accept", seller_offer
        
        if self._should_walk_away(seller_offer):
            return "reject", 0
        
        # Counter offer strategy
        counter = self._calculate_counter_offer(seller_offer, seller_style)
        return "offer", min(counter, self.budget)
    
    def _should_accept(self, offer: int, seller_style: str) -> bool:
        """Determine if offer should be accepted"""
        if offer > self.budget:
            return False
        
        # Different acceptance criteria based on seller style and round
        if seller_style == "aggressive" and self.round_num >= 7:
            return offer <= self.budget * 0.95
        elif seller_style == "diplomatic" and self.round_num >= 5:
            return offer <= self.product.base_market_price * 0.9
        else:
            return offer <= self.product.base_market_price * 0.85
    
    def _should_walk_away(self, offer: int) -> bool:
        """Determine if should walk away from negotiation"""
        if offer > self.budget * 1.15:
            return True
        if self.round_num >= 9 and offer > self.budget:
            return True
        return False
    
    def _calculate_counter_offer(self, seller_offer: int, seller_style: str) -> int:
        """Calculate strategic counter offer"""
        # Different counter strategies based on seller style
        if seller_style == "aggressive":
            return int(seller_offer * 0.88)
        elif seller_style == "skillful_enthusiast":
            return int(seller_offer * 0.92)
        else:
            return int(seller_offer * 0.90)
    
    def make_pre_act_value(self) -> str:
        """Return strategy context for LLM"""
        product_name = self.product.name if self.product else "Unknown"
        return f"Round: {self.round_num}, Budget: ₹{self.budget:,}, Product: {product_name}"
    
    def get_state(self):
        """Return current state for serialization"""
        return {
            'budget': self.budget,
            'round_num': self.round_num,
            'product': self.product.__dict__ if self.product else None
        }
    
    def set_state(self, state):
        """Restore from saved state"""
        self.budget = state['budget']
        self.round_num = state['round_num']
        if state['product']:
            self.product = Product(**state['product'])

class MarketAnalystBuyerAgent:
    """
    Market Analyst Buyer Agent - Concordia Implementation
    A strategic negotiator that uses data and adaptation to counter any seller persona.
    """
    
    def __init__(self, name: str = "MarketAnalyst", model: language_model.LanguageModel = None):
        self.name = name
        self.model = model or OllamaLanguageModel()
        self._build_components()
    
    def _build_components(self):
        """Build required Concordia components"""
        self.personality = BuyerPersonalityComponent()
        self.memory = NegotiationMemoryComponent()
        self.observation = SellerObservationComponent()
        self.strategy = StrategicDecisionComponent()
        
        self.components = [
            self.personality,
            self.memory,
            self.observation,
            self.strategy
        ]
    
    def negotiate(self, product: Product, budget: int, seller_message: str) -> NegotiationResponse:
        """
        Main negotiation method
        
        Args:
            product: Product being negotiated
            budget: Your maximum budget (NEVER exceed)
            seller_message: Latest message from seller
            
        Returns:
            NegotiationResponse with your action and message
        """
        # Initialize strategy
        self.strategy.initialize_negotiation(product, budget)
        self.strategy.increment_round()
        
        # Extract offer from seller message
        seller_offer = self._extract_offer(seller_message)
        self.observation.update_observation(seller_message, seller_offer)
        self.memory.add_interaction("seller", seller_message, seller_offer)
        
        # Make strategic decision
        decision, offer = self.strategy.make_decision(
            seller_offer, 
            self.observation.seller_style
        )
        
        # Generate response message using LLM
        message = self._generate_response(decision, offer, seller_message, seller_offer)
        self.memory.add_interaction("buyer", message, offer)
        
        return NegotiationResponse(decision, offer, message)
    
    def _extract_offer(self, message: str) -> Optional[int]:
        """Extract numeric offer from message"""
        import re
        numbers = re.findall(r'₹?(\d{1,3}(?:,\d{3})*)', message)
        if numbers:
            return int(numbers[-1].replace(',', ''))
        return None
    
    def _generate_response(self, decision: str, offer: int, seller_message: str, seller_offer: Optional[int]) -> str:
        """Generate context-aware response message using LLM"""
        prompt = f"""
        {self.personality.make_pre_act_value()}
        {self.memory.make_pre_act_value()}
        {self.observation.make_pre_act_value()}
        {self.strategy.make_pre_act_value()}
        
        Seller's latest message: "{seller_message}"
        Your decision: {decision} at ₹{offer}
        
        Generate a professional, data-driven response that matches your Market Analyst personality.
        Keep it under 100 words and include relevant market references.
        """
        
        return self.model.sample_text(prompt)
    
    def get_state(self):
        """Get current state of all components"""
        return {
            'personality': self.personality.get_state(),
            'memory': self.memory.get_state(),
            'observation': self.observation.get_state(),
            'strategy': self.strategy.get_state()
        }
    
    def set_state(self, state):
        """Set state from saved data"""
        self.personality.set_state(state['personality'])
        self.memory.set_state(state['memory'])
        self.observation.set_state(state['observation'])
        self.strategy.set_state(state['strategy'])

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = MarketAnalystBuyerAgent()
    
    # Example product
    product = Product(
        name="Alphonso Mangoes",
        category="Mangoes",
        quantity=100,
        quality_grade="A",
        origin="Ratnagiri",
        base_market_price=180000,
        attributes={"ripeness": "optimal", "export_grade": True}
    )
    
    # Test negotiation
    response = agent.negotiate(
        product=product,
        budget=200000,
        seller_message="Premium Alphonso mangoes from Ratnagiri! Best quality at ₹250000"
    )
    
    print(f"Decision: {response.status}")
    print(f"Offer: ₹{response.offer:,}")
    print(f"Message: {response.message}")