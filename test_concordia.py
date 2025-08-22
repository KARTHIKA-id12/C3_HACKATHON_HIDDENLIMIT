#!/usr/bin/env python3
"""
Test script to verify Concordia installation and setup
Run this first before running the main buyer agent
"""

def test_concordia_installation():
    """Test if Concordia is properly installed"""
    print("🧪 Testing Concordia Installation...")
    
    try:
        # Test core imports
        from concordia.components import agent as agent_components
        from concordia.associative_memory import associative_memory
        from concordia.language_model import language_model
        print("✅ Core Concordia components imported successfully")
        
        # Test component creation
        class TestComponent(agent_components.ContextComponent):
            def __init__(self):
                super().__init__("TestComponent")
            
            def make_pre_act_value(self):
                return "Test component active"
        
        test_component = TestComponent()
        print("✅ Concordia components can be created")
        
        # Test component functionality
        result = test_component.make_pre_act_value()
        assert result == "Test component active"
        print("✅ Concordia components functioning correctly")
        
        print("\n🎉 CONCORDIA INSTALLATION SUCCESSFUL!")
        print("✅ Ready to run Market Analyst Buyer Agent")
        return True
        
    except ImportError as e:
        print(f"❌ Concordia Import Error: {e}")
        print("\n🚨 INSTALLATION REQUIRED:")
        print("1. Activate your virtual environment")
        print("2. Run: pip install git+https://github.com/google-deepmind/concordia.git")
        print("3. Install dependencies: pip install numpy pandas typing-extensions")
        return False
        
    except Exception as e:
        print(f"❌ Concordia Setup Error: {e}")
        print("🔧 Check your installation and dependencies")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("\n📋 Testing Dependencies...")
    
    required_packages = [
        ("numpy", "1.26.0"),
        ("pandas", "1.3.0"),
        ("typing_extensions", "4.0.0")
    ]
    
    for package, min_version in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            print(f"   Install with: pip install {package}>={min_version}")

def test_language_model_setup():
    """Test language model configuration options"""
    print("\n🤖 Testing Language Model Options...")
    
    # Test OpenAI
    try:
        import openai
        print("✅ OpenAI available (set OPENAI_API_KEY environment variable)")
    except ImportError:
        print("⚠️  OpenAI not available (pip install openai)")
    
    # Test HuggingFace
    try:
        import transformers
        print("✅ HuggingFace Transformers available")
    except ImportError:
        print("⚠️  HuggingFace not available (pip install transformers torch)")
    
    # Test Ollama
    try:
        import ollama
        print("✅ Ollama available (install from https://ollama.ai)")
    except ImportError:
        print("⚠️  Ollama not available (pip install ollama)")

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 CONCORDIA INSTALLATION TEST")
    print("=" * 60)
    
    # Run tests
    concordia_ok = test_concordia_installation()
    test_dependencies()
    test_language_model_setup()
    
    print("\n" + "=" * 60)
    if concordia_ok:
        print("🚀 SYSTEM READY - You can now run your Market Analyst Agent!")
        print("   Run: python buyer_agent.py")
    else:
        print("🛠️  SETUP REQUIRED - Follow installation steps above")
    print("=" * 60)