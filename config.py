import os
import json
from pathlib import Path

# 配置文件路径
CONFIG_PATH = Path(__file__).parent / "config.json"

# 默认配置
DEFAULT_CONFIG = {
    "llm_models": {
        "openai": {
            "api_key": "",
            "model": "gpt-3.5-turbo",
            "api_base": ""  # 添加API基础URL字段
        }
    }
}

def load_config():
    """加载配置文件"""
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(config):
    """保存配置文件"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def update_model_config(provider, api_key, model_name, api_base=""):
    """更新模型配置"""
    config = load_config()
    
    if provider not in config["llm_models"]:
        config["llm_models"][provider] = {}
    
    config["llm_models"][provider]["api_key"] = api_key
    config["llm_models"][provider]["model"] = model_name
    config["llm_models"][provider]["api_base"] = api_base  # 更新API基础URL
    
    save_config(config)
    return config

def get_model_config(provider):
    """获取模型配置"""
    config = load_config()
    return config["llm_models"].get(provider, {})