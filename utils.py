import pandas as pd
import numpy as np
import io
import openai
from openai import OpenAI

from config import get_model_config

def load_data(file):
    """加载数据文件（CSV或TXT）"""
    if file is None:
        return None

    file_extension = file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension == 'txt':
            # 尝试不同的分隔符
            for sep in [',', '\t', ' ', ';']:
                try:
                    df = pd.read_csv(file, sep=sep)
                    # 如果只有一列，可能分隔符不正确
                    if len(df.columns) > 1:
                        break
                except:
                    continue
        else:
            return None, "不支持的文件格式，请上传CSV或TXT文件"

        return df, None
    except Exception as e:
        return None, f"加载文件时出错: {str(e)}"

def analyze_data(df):
    """分析数据基本信息"""
    if df is None:
        return {}

    # 基本信息
    info = {
        "行数": len(df),
        "列数": len(df.columns),
        "列名": list(df.columns),
        "数据类型": {col: str(df[col].dtype) for col in df.columns},
        "缺失值": {col: int(df[col].isna().sum()) for col in df.columns},
        "唯一值数量": {col: int(df[col].nunique()) for col in df.columns}
    }

    # 数值列的统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info["数值统计"] = {}
        for col in numeric_cols:
            info["数值统计"][col] = {
                "最小值": float(df[col].min()),
                "最大值": float(df[col].max()),
                "平均值": float(df[col].mean()),
                "中位数": float(df[col].median()),
                "标准差": float(df[col].std())
            }

    return info

def get_llm_suggestion(df_info, provider="openai"):
    """使用大语言模型获取数据可视化和模型建议"""
    config = get_model_config(provider)

    if not config or not config.get("api_key"):
        return "请先在设置中配置API密钥"

    api_key = config["api_key"]
    model = config.get("model", "gpt-3.5-turbo")
    api_base = config.get("api_base", None)  # 获取API基础URL

    # 设置API密钥和基础URL
    if api_base:
        # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=api_base)'
        # openai.api_base = api_base
        pass

    # 构建提示
    prompt = f"""
    我有一个数据集，以下是数据集的基本信息:
    {df_info}
    
    请根据这些信息，回答以下问题:
    1. 推荐3-5种适合这个数据集的可视化方式，并说明原因
    2. 根据数据特征，推荐2-3种适合的机器学习模型，并说明为什么这些模型适合这个数据集
    3. 对于每个推荐的机器学习模型，简要说明其优缺点和适用场景
    
    请以JSON格式回答，格式如下:
    {{
        "visualizations": [
            {{"name": "可视化名称", "reason": "推荐原因", "type": "图表类型"}}
        ],
        "models": [
            {{"name": "模型名称", "reason": "推荐原因", "pros": "优点", "cons": "缺点", "use_case": "适用场景"}}
        ]
    }}
    """

    try:
        # 在函数内部初始化客户端
        # 确保API基础URL不包含路径部分
        if api_base and '/v1/chat/completions' in api_base:
            # 如果API基础URL包含路径，则去除路径部分
            api_base = api_base.split('/v1/')[0]
        
        client = OpenAI(api_key=api_key, base_url=api_base if api_base else None)
        
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "system", "content": "你是一个数据科学专家，擅长数据分析和机器学习模型选择。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5)

        return response.choices[0].message.content
    except Exception as e:
        return f"调用LLM时出错: {str(e)}"

def identify_target_column(df):
    """识别可能的目标列"""
    # 如果列数少于2，无法进行预测
    if len(df.columns) < 2:
        return None

    # 优先选择数值型列作为目标
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # 选择最后一列作为目标列（常见约定）
        return numeric_cols[-1]

    # 如果没有数值型列，选择分类列
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        return categorical_cols[-1]

    # 默认选择最后一列
    return df.columns[-1]