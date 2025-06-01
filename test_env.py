import os

def test_env_variables():
    openai_key = os.getenv("OPENAI_API_KEY")
    newsdata_key = os.getenv("NEWSDATA_API_KEY")
    
    print("OpenAI API Key:", "✓ Present" if openai_key else "✗ Missing")
    print("NewsData API Key:", "✓ Present" if newsdata_key else "✗ Missing")

if __name__ == "__main__":
    test_env_variables()