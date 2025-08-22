#!/usr/bin/env python3
"""
Test script for Market Analyst Buyer Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from buyer_agent import YourBuyerAgent, Product, NegotiationResponse
from concordia.language_model import language_model
import random

class MockLanguageModel(language_model.LanguageModel):
    """Mock language model for testing"""
    
    def sample_text(self, prompt: str, max_tokens: int = 100) -> str:
        # Simple response generation for testing
        if "accept" in prompt.lower():
            return "I accept this offer based on market data analysis."
        elif "reject" in prompt.lower():
            return "I must decline as this doesn't align with market realities."
        else:
            return "Based on my market research, I propose this counter-offer."

def test_scenarios():
    """Test the agent against different scenarios"""
    
    # Initialize agent
    model = MockLanguageModel()
    agent = YourBuyerAgent("MarketAnalyst", "analytical_adaptive", model)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Easy Market - Alphonso Mangoes",
            "product": Product(
                name="Alphonso Mangoes", category="Mangoes", quantity=100,
                quality_grade="A", origin="Ratnagiri", base_market_price=180000,
                attributes={"ripeness": "optimal", "export_grade": True}
            ),
            "budget": 200000,
            "seller_messages": [
                "Premium Alphonso mangoes from Ratnagiri! Best quality at ₹250000",
                "I can offer ₹230000 for these export-grade mangoes",
                "Final offer: ₹210000 - they're selling fast!"
            ]
        },
        {
            "name": "Tight Budget - Kesar Mangoes", 
            "product": Product(
                name="Kesar Mangoes", category="Mangoes", quantity=150,
                quality_grade="B", origin="Gujarat", base_market_price=150000,
                attributes={"ripeness": "semi-ripe", "export_grade": False}
            ),
            "budget": 140000,
            "seller_messages": [
                "Kesar mangoes at ₹190000 - good quantity available",
                "Price drop to ₹175000 for quick sale",
                "Best I can do: ₹160000"
            ]
        }
    ]
    
    print("Testing Market Analyst Buyer Agent")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\n🔹 {scenario['name']}")
        print(f"   Budget: ₹{scenario['budget']:,}")
        print(f"   Market Price: ₹{scenario['product'].base_market_price:,}")
        
        for i, seller_msg in enumerate(scenario['seller_messages']):
            response = agent.negotiate(scenario['product'], scenario['budget'], seller_msg)
            print(f"   Round {i+1}: {response.status} at ₹{response.offer:,}")
            print(f"   Message: {response.message}")
            
            if response.status != "offer":
                break

if __name__ == "__main__":
    test_scenarios()