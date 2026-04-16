"""环境测试脚本"""
import sys

def test_imports():
    """测试关键库是否安装成功"""
    tests = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "streamlit": "Streamlit",
        "neo4j": "Neo4j Driver",
        "pandas": "Pandas",
        "jieba": "Jieba"
    }
    
    print("=" * 50)
    print("环境测试")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print("-" * 50)
    
    for module, name in tests.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - 安装成功")
        except ImportError:
            print(f"❌ {name:20s} - 未安装")
    
    print("=" * 50)

if __name__ == "__main__":
    test_imports()
