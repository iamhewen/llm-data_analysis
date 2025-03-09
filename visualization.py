import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import json

def create_visualization(df, viz_type, x_col=None, y_col=None, color_col=None):
    """创建可视化图表"""
    if df is None or len(df) == 0:
        return None
    
    try:
        if viz_type == "散点图":
            return create_scatter_plot(df, x_col, y_col, color_col)
        elif viz_type == "折线图":
            return create_line_chart(df, x_col, y_col, color_col)
        elif viz_type == "柱状图":
            return create_bar_chart(df, x_col, y_col, color_col)
        elif viz_type == "直方图":
            return create_histogram(df, x_col)
        elif viz_type == "箱线图":
            return create_box_plot(df, x_col, y_col)
        elif viz_type == "热力图":
            return create_heatmap(df)
        elif viz_type == "饼图":
            return create_pie_chart(df, x_col, y_col)
        elif viz_type == "相关性矩阵":
            return create_correlation_matrix(df)
        else:
            return None
    except Exception as e:
        st.error(f"创建可视化时出错: {str(e)}")
        return None

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """创建散点图"""
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
    return fig

def create_line_chart(df, x_col, y_col, color_col=None):
    """创建折线图"""
    if color_col and color_col in df.columns:
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} 随 {x_col} 的变化")
    else:
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} 随 {x_col} 的变化")
    return fig

def create_bar_chart(df, x_col, y_col, color_col=None):
    """创建柱状图"""
    if color_col and color_col in df.columns:
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} 的 {y_col} 分布")
    else:
        fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} 的 {y_col} 分布")
    return fig

def create_histogram(df, x_col):
    """创建直方图"""
    fig = px.histogram(df, x=x_col, title=f"{x_col} 的分布")
    return fig

def create_box_plot(df, x_col, y_col):
    """创建箱线图"""
    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} 按 {x_col} 分组的箱线图")
    return fig

def create_heatmap(df):
    """创建热力图（相关性矩阵）"""
    # 只选择数值列
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        colorbar=dict(title="相关系数")
    ))
    fig.update_layout(title="相关性矩阵")
    return fig

def create_pie_chart(df, x_col, y_col=None):
    """创建饼图"""
    if y_col:
        # 如果提供了y_col，按x_col分组并聚合y_col
        values = df.groupby(x_col)[y_col].sum()
        fig = px.pie(values=values, names=values.index, title=f"{x_col} 的 {y_col} 分布")
    else:
        # 否则，计算x_col的值计数
        value_counts = df[x_col].value_counts()
        fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"{x_col} 的分布")
    return fig

def create_correlation_matrix(df):
    """创建相关性矩阵"""
    # 只选择数值列
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) < 2:
        return None
    
    corr = numeric_df.corr()
    fig = px.imshow(corr, 
                   labels=dict(x="特征", y="特征", color="相关系数"),
                   x=corr.columns,
                   y=corr.columns,
                   color_continuous_scale="RdBu_r")
    fig.update_layout(title="相关性矩阵")
    return fig

def parse_llm_visualization_suggestions(llm_response):
    """解析LLM返回的可视化建议"""
    try:
        # 尝试直接解析JSON
        data = json.loads(llm_response)
        return data.get("visualizations", [])
    except:
        # 如果解析失败，尝试从文本中提取JSON部分
        try:
            json_str = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
            data = json.loads(json_str)
            return data.get("visualizations", [])
        except:
            # 如果仍然失败，返回空列表
            return []

def parse_llm_model_suggestions(llm_response):
    """解析LLM返回的模型建议"""
    try:
        # 尝试直接解析JSON
        data = json.loads(llm_response)
        return data.get("models", [])
    except:
        # 如果解析失败，尝试从文本中提取JSON部分
        try:
            json_str = llm_response[llm_response.find('{'):llm_response.rfind('}')+1]
            data = json.loads(json_str)
            return data.get("models", [])
        except:
            # 如果仍然失败，返回空列表
            return []

def get_recommended_visualizations(df):
    """根据数据特征自动推荐可视化类型"""
    recommendations = []
    
    # 检查数据集大小
    if df is None or len(df) == 0:
        return recommendations
    
    # 获取数值列和分类列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 如果有两个或以上数值列，推荐散点图和相关性矩阵
    if len(numeric_cols) >= 2:
        recommendations.append({
            "name": "散点图",
            "reason": "适合展示两个数值变量之间的关系",
            "type": "scatter"
        })
        recommendations.append({
            "name": "相关性矩阵",
            "reason": "适合展示所有数值变量之间的相关性",
            "type": "correlation"
        })
    
    # 如果有数值列，推荐直方图和箱线图
    if len(numeric_cols) >= 1:
        recommendations.append({
            "name": "直方图",
            "reason": "适合展示单个数值变量的分布",
            "type": "histogram"
        })
        
        if len(categorical_cols) >= 1:
            recommendations.append({
                "name": "箱线图",
                "reason": "适合比较不同类别下数值变量的分布",
                "type": "box"
            })
    
    # 如果有分类列，推荐饼图和柱状图
    if len(categorical_cols) >= 1:
        recommendations.append({
            "name": "饼图",
            "reason": "适合展示分类变量的比例分布",
            "type": "pie"
        })
        recommendations.append({
            "name": "柱状图",
            "reason": "适合比较不同类别的数量或其他指标",
            "type": "bar"
        })
    
    # 如果行数较多且有数值列，推荐折线图
    if len(df) > 10 and len(numeric_cols) >= 1:
        recommendations.append({
            "name": "折线图",
            "reason": "适合展示数据随时间或顺序的变化趋势",
            "type": "line"
        })
    
    return recommendations