# 数据分析工具使用手册

## 环境要求
- Python 3.8+ 
- macOS/Linux/Windows（推荐macOS）

### 特殊依赖
- **Mac用户注意**：本项目使用的XGBoost和LightGBM需要OpenMP运行时库
  ```bash
  brew install libomp
  ```
  如果遇到"Library not loaded: @rpath/libomp.dylib"错误，请执行上述命令安装依赖

## 安装步骤
1. 克隆仓库
```bash
git clone https://github.com/iamhewen/llm-data_analysis.git
cd Data_analysis
```
2. 安装依赖
```bash
pip install -r requirements.txt
```

## 快速启动
```bash
streamlit run app.py
```
访问提示的本地URL（通常为 http://localhost:8501）

## 核心功能
### 数据上传与分析
1. 通过左侧菜单上传CSV/TXT文件
2. 查看数据预览和统计信息
3. 点击「获取AI分析建议」获得可视化方案

### 模型训练
1. 在「模型训练与预测」页选择目标列
2. 从推荐模型中选择并调整参数
3. 查看训练结果和评估指标

### 设置
1. 配置OpenAI API密钥
2. 修改API基础URL（如需）
3. 选择大模型版本

## API密钥配置
1. 访问[OpenAI平台](https://platform.openai.com)获取API密钥
2. 在设置页面输入密钥并保存

## 常见问题
Q: 上传文件时报错「不支持的文件格式」
A: 请确保文件为CSV或TXT格式，且包含有效数据

Q: 模型训练时出现内存不足
A: 尝试减小数据集规模或选择更轻量级的模型

Q: 可视化图表无法显示
A: 检查是否选择了有效的列组合，数值列需包含有效数据

[//]: # (以下为自动生成的联系方式)
## 技术支持
联系邮箱：youwin_one@126.com