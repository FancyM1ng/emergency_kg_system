# 基于知识图谱的智能化应急管理辅助系统

## 项目简介
本项目结合知识图谱与大语言模型技术，构建智能化的应急管理辅助系统。

## 快速开始

### 1. 克隆项目
\`\`\`bash
git clone <your-repo-url>
cd emergency_kg_system
\`\`\`

### 2. 创建虚拟环境
\`\`\`bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
\`\`\`

### 3. 安装依赖
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. 配置环境变量
\`\`\`bash
cp config/.env.example config/.env
# 编辑.env文件，填入你的配置
\`\`\`

### 5. 运行应用
\`\`\`bash
streamlit run app/streamlit_app.py
\`\`\`

## 项目结构
- `kg/` - 知识图谱模块
- `model/` - 模型模块
- `app/` - Web应用
- `utils/` - 工具函数
- `tests/` - 测试代码

## 技术栈
- Python 3.9+
- PyTorch
- Transformers
- Neo4j
- Streamlit

## 开发者
张可铭
