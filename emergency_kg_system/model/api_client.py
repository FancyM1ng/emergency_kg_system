"""智谱AI API客户端"""
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config" / ".env"

class ZhipuClient:
    """智谱AI客户端"""
    
    def __init__(self):
        """初始化"""
        load_dotenv(ENV_FILE)
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('ZHIPU_API_KEY')
        self.api_url = os.getenv('DEEPSEEK_API_URL', "https://api.deepseek.com/chat/completions")
        self.model = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")
        
        if not self.api_key:
            raise ValueError("未找到DEEPSEEK_API_KEY，请在config/.env中配置")
    
    def chat(self, messages, model=None):
        """调用聊天接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            
        Returns:
            模型回复
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model or self.model,
            "messages": messages
        }
        
        try:
            response = requests.post(self.api_url, 
                                    headers=headers, 
                                    json=data,
                                    timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            return f"API调用失败: {str(e)}"

# 测试代码
if __name__ == "__main__":
    client = ZhipuClient()
    
    messages = [
        {"role": "user", "content": "化工厂爆炸应该采取哪些应急措施？"}
    ]
    
    print("正在调用API...")
    answer = client.chat(messages)
    print(f"\n回答：\n{answer}")
