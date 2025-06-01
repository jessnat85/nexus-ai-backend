import os

# Try to load dotenv only if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ dotenv module not available. Using fallback values...")

# Fallback values in case .env isn't loaded or accessible
openai_key = os.getenv("OPENAI_API_KEY", "FAKE_OPENAI_KEY")
newsdata_key = os.getenv("NEWSDATA_API_KEY", "FAKE_NEWSDATA_KEY")

def test_env_variables():
    print("OpenAI API Key:", "✓ Present" if openai_key and not openai_key.startswith("FAKE") else "✗ Missing")
    print("NewsData API Key:", "✓ Present" if newsdata_key and not newsdata_key.startswith("FAKE") else "✗ Missing")

if __name__ == "__main__":
    test_env_variables()