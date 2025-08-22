"""
===========================================
AI NEGOTIATION AGENT - INTERVIEW TEMPLATE
===========================================

"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import random

# ============================================
# PART 1: DATA STRUCTURES (DO NOT MODIFY)
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_budget: int  # Your maximum budget (NEVER exceed this)
    current_round: int
    seller_offers: List[int]  # History of seller's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


# ============================================
# PART 2: BASE AGENT CLASS (DO NOT MODIFY)
# ============================================

class BaseBuyerAgent(ABC):
    """Base class for all buyer agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
        
    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        """
        Define your agent's personality traits.
        
        Returns:
            Dict containing:
            - personality_type: str (e.g., "aggressive", "analytical", "diplomatic", "custom")
            - traits: List[str] (e.g., ["impatient", "data-driven", "friendly"])
            - negotiation_style: str (description of approach)
            - catchphrases: List[str] (typical phrases your agent uses)
        """
        pass
    
    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        Generate your first offer in the negotiation.
        
        Args:
            context: Current negotiation context
            
        Returns:
            Tuple of (offer_amount, message)
            - offer_amount: Your opening price offer (must be <= budget)
            - message: Your negotiation message (2-3 sentences, include personality)
        """
        pass
    
    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """
        Respond to the seller's offer.
        
        Args:
            context: Current negotiation context
            seller_price: The seller's current price offer
            seller_message: The seller's message
            
        Returns:
            Tuple of (deal_status, counter_offer, message)
            - deal_status: ACCEPTED if you take the deal, ONGOING if negotiating
            - counter_offer: Your counter price (ignored if deal_status is ACCEPTED)
            - message: Your response message
        """
        pass
    
    @abstractmethod
    def get_personality_prompt(self) -> str:
        """
        Return a prompt that describes how your agent should communicate.
        This will be used to evaluate character consistency.
        
        Returns:
            A detailed prompt describing your agent's communication style
        """
        pass


# ============================================
# PART 3: YOUR IMPLEMENTATION STARTS HERE
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    MARKET ANALYST BUYER AGENT
    
    A data-driven negotiator that counters any seller persona with market analysis
    and strategic adaptation while maintaining strict budget discipline.
    """
    
    def define_personality(self) -> Dict[str, Any]:
        """
        Market Analyst personality - versatile against any seller type
        """
        return {
            "personality_type": "analytical_adaptive",
            "traits": ["data-driven", "strategic", "verification-focused", "budget-aware", "adaptable"],
            "negotiation_style": "Uses market data and competitor analysis to counter any seller tactic. Verifies all claims with factual research and maintains firm budget boundaries while adapting to different seller styles.",
            "catchphrases": [
                "My market verification shows...",
                "Let's examine the actual data on...", 
                "Competitor benchmarks indicate...",
                "The numbers suggest a fair value would be...",
                "I need to validate those claims against market reality"
            ]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        """
        Generate strategic opening offer based on comprehensive market analysis
        """
        # Calculate quality-adjusted market price
        quality_factor = self._calculate_quality_factor(context.product)
        
        # Start at 65% of adjusted market price for negotiation room
        base_offer = int(context.product.base_market_price * 0.65 * quality_factor)
        
        # Apply psychological pricing (charm pricing)
        opening_price = self._apply_charm_pricing(base_offer)
        
        # Ensure within budget and reasonable minimum
        opening_price = min(opening_price, context.your_budget)
        opening_price = max(opening_price, int(context.product.base_market_price * 0.55))
        
        message = (f"My market research shows {context.product.quality_grade} grade {context.product.name} "
                  f"from {context.product.origin} are trading at ₹{context.product.base_market_price:,} "
                  f"for this quantity. Competitors offer similar quality at 15-20% lower. "
                  f"I'll open with ₹{opening_price:,} based on current market data.")
        
        return opening_price, message
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        """
        Analyze seller's offer and respond with strategic counter-tactics
        """
        # Analyze negotiation progress
        progress = self._analyze_negotiation_progress(context, seller_price)
        
        # Identify seller's tactics from their message
        seller_tactics = self._identify_seller_tactics(seller_message)
        
        # Check if we should accept
        if self._should_accept_offer(context, seller_price, progress, seller_tactics):
            message = self._generate_acceptance_message(seller_price, context)
            return DealStatus.ACCEPTED, seller_price, message
        
        # Check if we should walk away
        if self._should_walk_away(context, seller_price, progress):
            message = self._generate_walkaway_message(context)
            return DealStatus.REJECTED, 0, message
        
        # Generate strategic counter offer
        counter_offer = self._calculate_strategic_counter(context, seller_price, progress, seller_tactics)
        counter_offer = min(counter_offer, context.your_budget)  # Never exceed budget
        
        # Craft tactical response message
        message = self._generate_tactical_response(context, seller_price, seller_message, counter_offer, seller_tactics)
        
        return DealStatus.ONGOING, counter_offer, message
    
    def get_personality_prompt(self) -> str:
        """
        Detailed prompt for consistent communication style
        """
        return """
        I am a Market Analyst buyer who uses data-driven negotiation strategies. 
        I counter premium pricing claims with market research and competitor analysis.
        I verify quality assertions and question value propositions that don't align with market rates.
        My tone is professional but firm, and I emphasize budget constraints without revealing exact limits.
        I use psychological pricing tactics and strategic concessions based on negotiation progress.
        I maintain a fact-focused, analytical approach throughout the negotiation.
        I adapt my responses based on the seller's style while staying true to my data-driven personality.
        """

    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _calculate_quality_factor(self, product: Product) -> float:
        """Calculate quality adjustment based on objective metrics"""
        quality_factors = {
            "Export": 1.1,
            "A": 1.0,
            "B": 0.85
        }
        base_factor = quality_factors.get(product.quality_grade, 0.9)
        
        # Adjust for specific attributes
        if product.attributes.get("export_grade", False):
            base_factor *= 1.05
        if product.attributes.get("ripeness") == "optimal":
            base_factor *= 1.03
            
        return min(base_factor, 1.15)  # Cap premium factors
    
    def _apply_charm_pricing(self, price: int) -> int:
        """Apply psychological pricing tactics"""
        # Round to nearest 999 or 499 for psychological effect
        if price > 100000:
            return (price // 1000) * 1000 - 1  # ₹199,999 instead of ₹200,000
        else:
            return (price // 100) * 100 - 1  # ₹19,999 instead of ₹20,000
    
    def _identify_seller_tactics(self, seller_message: str) -> List[str]:
        """Identify the tactical approaches used by the seller"""
        tactics = []
        message_lower = seller_message.lower()
        
        tactic_patterns = {
            "quality_emphasis": ["premium", "quality", "export", "grade", "finest"],
            "rarity_scarcity": ["rare", "limited", "scarcity", "exclusive", "last"],
            "cost_justification": ["shipping", "labor", "cost", "expense", "investment"],
            "urgency_creation": ["now", "today", "quick", "urgent", "opportunity"],
            "credibility_building": ["certified", "award", "testimonial", "reputation"]
        }
        
        for tactic, keywords in tactic_patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                tactics.append(tactic)
                
        return tactics
    
    def _analyze_negotiation_progress(self, context: NegotiationContext, current_offer: int) -> Dict[str, Any]:
        """Comprehensive analysis of negotiation progress"""
        if not context.seller_offers:
            return {
                "rounds_remaining": 10 - context.current_round,
                "seller_movement": 0,
                "concession_rate": 0,
                "leverage_score": 0.5
            }
            
        initial_offer = context.seller_offers[0]
        seller_movement = (initial_offer - current_offer) / initial_offer if initial_offer > 0 else 0
        
        # Calculate concession patterns
        concession_pattern = self._analyze_concession_pattern(context.seller_offers)
        
        # Assess our negotiation leverage
        leverage_score = self._calculate_leverage_score(context, current_offer)
        
        return {
            "rounds_remaining": 10 - context.current_round,
            "seller_movement": seller_movement,
            "concession_rate": concession_pattern["rate"],
            "leverage_score": leverage_score,
            "seller_pattern": concession_pattern["pattern"]
        }
    
    def _analyze_concession_pattern(self, seller_offers: List[int]) -> Dict[str, Any]:
        """Analyze the seller's concession pattern"""
        if len(seller_offers) < 2:
            return {"rate": 0, "pattern": "unknown"}
        
        concessions = []
        for i in range(1, len(seller_offers)):
            if seller_offers[i-1] > 0:
                concession_pct = (seller_offers[i-1] - seller_offers[i]) / seller_offers[i-1]
                concessions.append(concession_pct)
        
        avg_rate = sum(concessions) / len(concessions) if concessions else 0
        
        # Identify pattern
        if len(concessions) >= 3 and concessions[-1] > concessions[-2] > concessions[-3]:
            pattern = "accelerating"
        elif all(0.05 <= rate <= 0.1 for rate in concessions):
            pattern = "steady"
        elif any(rate > 0.15 for rate in concessions):
            pattern = "erratic"
        else:
            pattern = "conservative"
            
        return {"rate": avg_rate, "pattern": pattern}
    
    def _calculate_leverage_score(self, context: NegotiationContext, current_offer: int) -> float:
        """Calculate our negotiation leverage score (0-1)"""
        if context.your_budget == 0:
            return 0.5
            
        budget_utilization = current_offer / context.your_budget
        rounds_passed = context.current_round / 10
        
        # More leverage when we have budget room and time
        leverage = (1 - budget_utilization) * (1 - rounds_passed)
        return max(0, min(1, leverage))
    
    def _should_accept_offer(self, context: NegotiationContext, offer: int, 
                           progress: Dict[str, Any], seller_tactics: List[str]) -> bool:
        """Strategic decision to accept an offer"""
        if offer > context.your_budget:
            return False
        
        # Always accept if within budget and below market
        if offer <= context.product.base_market_price:
            return True
        
        # Accept if we're in final rounds and offer is reasonable
        if progress["rounds_remaining"] <= 2 and offer <= context.your_budget:
            return True
        
        # Accept if seller shows significant movement
        if (progress["seller_movement"] > 0.25 and 
            offer <= context.your_budget and 
            "cost_justification" not in seller_tactics):
            return True
            
        return False
    
    def _should_walk_away(self, context: NegotiationContext, offer: int, progress: Dict[str, Any]) -> bool:
        """Strategic decision to walk away"""
        # Walk away if significantly over budget
        if offer > context.your_budget * 1.15:
            return True
        
        # Walk away if no progress and we're late in negotiation
        if (progress["seller_movement"] < 0.05 and 
            progress["rounds_remaining"] <= 3 and 
            offer > context.your_budget * 0.95):
            return True
            
        return False
    
    def _calculate_strategic_counter(self, context: NegotiationContext, seller_offer: int, 
                                   progress: Dict[str, Any], seller_tactics: List[str]) -> int:
        """Calculate strategic counter offer based on multiple factors"""
        if context.your_offers:
            last_offer = context.your_offers[-1]
        else:
            last_offer = int(context.product.base_market_price * 0.65)
        
        # Base concession strategy
        if progress["rounds_remaining"] <= 2:
            # Final rounds: more aggressive movement
            concession = min(int(seller_offer * 0.93), context.your_budget)
        elif progress["seller_pattern"] == "accelerating":
            # Seller is accelerating concessions - move slowly
            concession = min(int(last_offer * 1.03), context.your_budget)
        else:
            # Standard concession
            concession = min(int(last_offer * 1.07), context.your_budget)
        
        # Adjust for seller tactics
        if "urgency_creation" in seller_tactics:
            # Seller creating urgency - move slower
            concession = min(int(concession * 0.95), context.your_budget)
        
        # Apply charm pricing
        concession = self._apply_charm_pricing(concession)
        
        return max(concession, last_offer)  # Never go backward
    
    def _generate_tactical_response(self, context: NegotiationContext, seller_price: int, 
                                  seller_message: str, counter_offer: int, seller_tactics: List[str]) -> str:
        """Generate tactical response countering seller's arguments"""
        response_templates = {
            "quality_emphasis": [
                "I appreciate the quality claims, but my market verification shows similar grade products at {} lower",
                "Your quality assertions need verification - competitors offer comparable specifications at {} less"
            ],
            "rarity_scarcity": [
                "While you mention scarcity, market data shows adequate supply with {} availability",
                "The scarcity argument doesn't align with current market inventory levels showing {} stock"
            ],
            "cost_justification": [
                "Your cost breakdown seems inflated - industry standards show {} lower expenses",
                "I've analyzed similar operations and your cost structure appears {} higher than market average"
            ],
            "urgency_creation": [
                "The urgency seems manufactured - my research indicates {} time availability",
                "Other suppliers aren't indicating the same urgency, suggesting {} timeframe flexibility"
            ]
        }
        
        # Select appropriate counter based on seller's primary tactic
        primary_tactic = seller_tactics[0] if seller_tactics else "general"
        templates = response_templates.get(primary_tactic, [
            "My data suggests a fair market value would be around {}",
            "Based on comprehensive analysis, {} represents appropriate market value"
        ])
        
        # Add budget pressure in later rounds
        budget_mention = ""
        if context.current_round >= 7:
            budget_mention = " This needs to work within my allocated budget constraints."
        
        # Format with price difference
        price_diff_pct = int((seller_price - counter_offer) / seller_price * 100) if seller_price > 0 else 20
        template = random.choice(templates)
        response = template.format(f"{price_diff_pct}%") + budget_mention
        
        return response
    
    def _generate_acceptance_message(self, price: int, context: NegotiationContext) -> str:
        """Generate professional acceptance message"""
        savings = context.your_budget - price
        return (f"After verifying the market data and quality specifications, "
                f"₹{price:,} represents fair value. I accept this offer, achieving "
                f"₹{savings:,} savings against my budget allocation.")
    
    def _generate_walkaway_message(self, context: NegotiationContext) -> str:
        """Generate professional walkaway message"""
        return (f"The numbers don't align with market realities and my budget constraints. "
                f"I'll need to pursue other opportunities that offer better value alignment.")

# ============================================
# PART 4: EXAMPLE SIMPLE AGENT (FOR REFERENCE)
# ============================================

class ExampleSimpleAgent(BaseBuyerAgent):
    """
    A simple example agent that you can use as reference.
    This agent has basic logic - you should do better!
    """
    
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "cautious",
            "traits": ["careful", "budget-conscious", "polite"],
            "negotiation_style": "Makes small incremental offers, very careful with money",
            "catchphrases": ["Let me think about that...", "That's a bit steep for me"]
        }
    
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        # Start at 60% of market price
        opening = int(context.product.base_market_price * 0.6)
        opening = min(opening, context.your_budget)
        
        return opening, f"I'm interested, but ₹{opening} is what I can offer. Let me think about that..."
    
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        # Accept if within budget and below 85% of market
        if seller_price <= context.your_budget and seller_price <= context.product.base_market_price * 0.85:
            return DealStatus.ACCEPTED, seller_price, f"Alright, ₹{seller_price} works for me!"
        
        # Counter with small increment
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = min(int(last_offer * 1.1), context.your_budget)
        
        if counter >= seller_price * 0.95:  # Close to agreement
            counter = min(seller_price - 1000, context.your_budget)
            return DealStatus.ONGOING, counter, f"That's a bit steep for me. How about ₹{counter}?"
        
        return DealStatus.ONGOING, counter, f"I can go up to ₹{counter}, but that's pushing my budget."
    
    def get_personality_prompt(self) -> str:
        return """
        I am a cautious buyer who is very careful with money. I speak politely but firmly.
        I often say things like 'Let me think about that' or 'That's a bit steep for me'.
        I make small incremental offers and show concern about my budget.
        """


# ============================================
# PART 5: TESTING FRAMEWORK (DO NOT MODIFY)
# ============================================

class MockSellerAgent:
    """A simple mock seller for testing your agent"""
    
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality
        
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        # Start at 150% of market price
        price = int(product.base_market_price * 1.5)
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."
    
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price * 1.1:  # Good profit
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
            
        if round_num >= 8:  # Close to timeout
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter}.", False


def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    """Test a negotiation between your buyer and a mock seller"""
    
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(
        product=product,
        your_budget=buyer_budget,
        current_round=0,
        seller_offers=[],
        your_offers=[],
        messages=[]
    )
    
    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})
    
    # Run negotiation
    deal_made = False
    final_price = None
    
    for round_num in range(10):  # Max 10 rounds
        context.current_round = round_num + 1
        
        # Buyer responds
        if round_num == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(
                context, seller_price, seller_msg
            )
        
        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})
        
        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break
            
        # Seller responds
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(buyer_offer, round_num)
        
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            break
            
        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})
    
    # Calculate results
    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "savings": buyer_budget - final_price if deal_made else 0,
        "savings_pct": ((buyer_budget - final_price) / buyer_budget * 100) if deal_made else 0,
        "below_market_pct": ((product.base_market_price - final_price) / product.base_market_price * 100) if deal_made else 0,
        "conversation": context.messages
    }
    
    return result


# ============================================
# PART 6: TEST YOUR AGENT
# ============================================

def test_your_agent():
    """Run this to test your agent implementation"""
    
    # Create test products
    test_products = [
        Product(
            name="Alphonso Mangoes",
            category="Mangoes",
            quantity=100,
            quality_grade="A",
            origin="Ratnagiri",
            base_market_price=180000,
            attributes={"ripeness": "optimal", "export_grade": True}
        ),
        Product(
            name="Kesar Mangoes", 
            category="Mangoes",
            quantity=150,
            quality_grade="B",
            origin="Gujarat",
            base_market_price=150000,
            attributes={"ripeness": "semi-ripe", "export_grade": False}
        )
    ]
    
    # Initialize your agent
    your_agent = YourBuyerAgent("TestBuyer")
    
    print("="*60)
    print(f"TESTING YOUR AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
    print("="*60)
    
    total_savings = 0
    deals_made = 0
    
    # Run multiple test scenarios
    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:  # hard
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)
            
            print(f"\nTest: {product.name} - {scenario} scenario")
            print(f"Your Budget: ₹{buyer_budget:,} | Market Price: ₹{product.base_market_price:,}")
            
            result = run_negotiation_test(your_agent, product, buyer_budget, seller_min)
            
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}%)")
                print(f"   Below Market: {result['below_market_pct']:.1f}%")
            else:
                print(f"❌ NO DEAL after {result['rounds']} rounds")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Savings: ₹{total_savings:,}")
    print(f"Success Rate: {deals_made/6*100:.1f}%")
    print("="*60)



if __name__ == "__main__":
    # Run this to test your implementation
    test_your_agent()
    
    # Uncomment to see how the example agent performs
    # print("\n\nTESTING EXAMPLE AGENT FOR COMPARISON:")
    # example_agent = ExampleSimpleAgent("ExampleBuyer")
    # test_your_agent()