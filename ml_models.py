import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import streamlit as st
import joblib
import os
from pathlib import Path

# 模型保存路径
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 支持的模型
REGRESSION_MODELS = {
    "线性回归": LinearRegression,
    "随机森林回归": RandomForestRegressor,
    "支持向量机回归": SVR,
    "K近邻回归": KNeighborsRegressor
}

CLASSIFICATION_MODELS = {
    "逻辑回归": LogisticRegression,
    "随机森林分类": RandomForestClassifier,
    "支持向量机分类": SVC,
    "K近邻分类": KNeighborsClassifier
}

def is_classification_task(y):
    """判断是分类任务还是回归任务"""
    # 如果目标变量是字符串类型，则为分类任务
    if pd.api.types.is_string_dtype(y):
        return True
    
    # 如果目标变量是数值型但唯一值较少，也视为分类任务
    if pd.api.types.is_numeric_dtype(y):
        unique_count = len(y.unique())
        if unique_count < 10:  # 阈值可调整
            return True
    
    return False

def preprocess_data(df, target_col):
    """预处理数据"""
    if target_col not in df.columns:
        return None, None, None, "目标列不存在"
    
    # 分离特征和目标
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 处理分类特征
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 处理目标变量（如果是分类任务）
    is_classification = is_classification_task(y)
    if is_classification:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, is_classification

def train_model(df, target_col, model_name, params=None):
    """训练模型"""
    # 预处理数据
    X_train, X_test, y_train, y_test, is_classification = preprocess_data(df, target_col)
    
    if X_train is None:
        return None, None, "数据预处理失败"
    
    # 选择模型
    if is_classification:
        if model_name not in CLASSIFICATION_MODELS:
            return None, None, f"不支持的分类模型: {model_name}"
        model_class = CLASSIFICATION_MODELS[model_name]
    else:
        if model_name not in REGRESSION_MODELS:
            return None, None, f"不支持的回归模型: {model_name}"
        model_class = REGRESSION_MODELS[model_name]
    
    # 创建模型
    if params:
        model = model_class(**params)
    else:
        model = model_class()
    
    # 创建管道（包含标准化）
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 评估模型
    y_pred = pipeline.predict(X_test)
    
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            "准确率": accuracy,
            "精确率": report['weighted avg']['precision'],
            "召回率": report['weighted avg']['recall'],
            "F1分数": report['weighted avg']['f1-score']
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {
            "均方误差": mse,
            "R²分数": r2
        }
    
    # 保存模型
    model_path = MODELS_DIR / f"{model_name}_{target_col}.joblib"
    joblib.dump(pipeline, model_path)
    
    return pipeline, metrics, None

def predict(model, input_data, df, target_col):
    """使用模型进行预测"""
    if model is None:
        return None, "模型未训练"
    
    # 获取特征列
    feature_cols = df.drop(columns=[target_col]).columns
    
    # 检查输入数据是否包含所有特征
    for col in feature_cols:
        if col not in input_data:
            return None, f"缺少特征: {col}"
    
    # 创建输入数据DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 处理分类特征
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col in input_df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))
    
    # 进行预测
    try:
        prediction = model.predict(input_df[feature_cols])
        
        # 如果是分类任务，获取原始类别标签
        if is_classification_task(df[target_col]):
            le = LabelEncoder()
            le.fit(df[target_col].astype(str))
            prediction = le.inverse_transform(prediction.astype(int))
        
        return prediction[0], None
    except Exception as e:
        return None, f"预测时出错: {str(e)}"

def get_model_suggestions(df, target_col):
    """根据数据特征推荐适合的模型"""
    if target_col not in df.columns:
        return []
    
    # 判断任务类型
    is_classification = is_classification_task(df[target_col])
    
    # 获取特征信息
    feature_count = len(df.columns) - 1
    sample_count = len(df)
    categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
    numeric_count = len(df.select_dtypes(include=[np.number]).columns) - (0 if is_classification_task(df[target_col]) else 1)
    
    suggestions = []
    
    if is_classification:
        # 分类任务推荐
        suggestions.append({
            "name": "随机森林分类",
            "reason": "适用于大多数分类问题，能处理高维特征，不易过拟合",
            "pros": "可处理分类和数值特征，不需要特征缩放，能评估特征重要性",
            "cons": "训练较慢，模型较大，解释性不如决策树",
            "use_case": "适合复杂的分类问题，特别是有多种类型特征时"
        })
        
        if sample_count > feature_count * 10:
            suggestions.append({
                "name": "逻辑回归",
                "reason": "简单高效，适合线性可分问题，提供概率输出",
                "pros": "训练快速，内存占用小，易于解释",
                "cons": "只能处理线性关系，对异常值敏感",
                "use_case": "适合二分类问题或需要概率输出的场景"
            })
        
        if numeric_count > 0:
            suggestions.append({
                "name": "支持向量机分类",
                "reason": "在高维空间表现良好，适合复杂决策边界",
                "pros": "在高维数据上表现好，内存效率高，对噪声有一定鲁棒性",
                "cons": "对参数敏感，训练慢，难以解释",
                "use_case": "适合特征数量大于样本数量的情况，或需要非线性决策边界时"
            })
        
        if sample_count < 10000:
            suggestions.append({
                "name": "K近邻分类",
                "reason": "简单直观，无需训练，适合小数据集",
                "pros": "简单易实现，无需训练，适合多分类",
                "cons": "预测慢，需要大量内存，对特征缩放敏感",
                "use_case": "适合小型数据集或作为基准模型"
            })
    else:
        # 回归任务推荐
        suggestions.append({
            "name": "随机森林回归",
            "reason": "适用于大多数回归问题，能处理非线性关系，不易过拟合",
            "pros": "可处理分类和数值特征，不需要特征缩放，能评估特征重要性",
            "cons": "训练较慢，模型较大，解释性不如线性模型",
            "use_case": "适合复杂的回归问题，特别是有多种类型特征时"
        })
        
        if numeric_count > 0:
            suggestions.append({
                "name": "线性回归",
                "reason": "简单高效，适合线性关系，易于解释",
                "pros": "训练快速，内存占用小，易于解释",
                "cons": "只能处理线性关系，对异常值敏感",
                "use_case": "适合简单的回归问题或需要解释性的场景"
            })
            
            suggestions.append({
                "name": "支持向量机回归",
                "reason": "在高维空间表现良好，适合复杂非线性关系",
                "pros": "在高维数据上表现好，对噪声有一定鲁棒性",
                "cons": "对参数敏感，训练慢，难以解释",
                "use_case": "适合特征数量大于样本数量的情况，或需要非线性关系建模时"
            })
        
        if sample_count < 10000:
            suggestions.append({
                "name": "K近邻回归",
                "reason": "简单直观，无需训练，适合小数据集",
                "pros": "简单易实现，无需训练，适合非线性关系",
                "cons": "预测慢，需要大量内存，对特征缩放敏感",
                "use_case": "适合小型数据集或作为基准模型"
            })
    
    return suggestions