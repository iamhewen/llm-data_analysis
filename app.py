import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

from utils import load_data, analyze_data, get_llm_suggestion
from visualization import (
    create_visualization, 
    parse_llm_visualization_suggestions,
    parse_llm_model_suggestions,
    get_recommended_visualizations
)
from ml_models import train_model, predict, get_model_suggestions
from config import load_config, update_model_config

# 设置页面
st.set_page_config(
    page_title="数据分析与机器学习应用",
    page_icon="📊",
    layout="wide"
)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# 侧边栏 - 导航
st.sidebar.title("导航")
page = st.sidebar.radio("选择页面", ["数据上传与分析", "模型训练与预测", "设置"])

# 数据上传与分析页面
if page == "数据上传与分析":
    st.title("数据上传与分析")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传CSV或TXT文件", type=["csv", "txt"])
    
    if uploaded_file is not None:
        # 加载数据
        df, error = load_data(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.session_state.data = df
            st.session_state.filename = uploaded_file.name
            st.success(f"成功加载文件: {uploaded_file.name}")
            
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            # 数据基本信息
            st.subheader("数据基本信息")
            info = analyze_data(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"行数: {info['行数']}")
                st.write(f"列数: {info['列数']}")
            
            with col2:
                st.write("列名: " + ", ".join(info['列名']))
            
            # 数据类型和缺失值
            st.subheader("数据类型和缺失值")
            type_missing_df = pd.DataFrame({
                "数据类型": info["数据类型"],
                "缺失值数量": info["缺失值"],
                "唯一值数量": info["唯一值数量"]
            })
            st.dataframe(type_missing_df)
            
            # 数值统计
            if "数值统计" in info:
                st.subheader("数值统计")
                for col, stats in info["数值统计"].items():
                    st.write(f"**{col}**")
                    stats_df = pd.DataFrame(stats, index=[0])
                    st.dataframe(stats_df)
            
            # 使用LLM分析数据
            st.subheader("数据分析与可视化建议")
            
            if st.button("获取AI分析建议"):
                with st.spinner("正在分析数据..."):
                    llm_response = get_llm_suggestion(info)
                    st.session_state.llm_response = llm_response
            
            if st.session_state.llm_response:
                # 显示原始响应
                with st.expander("查看AI完整分析"):
                    st.write(st.session_state.llm_response)
                
                # 解析可视化建议
                viz_suggestions = parse_llm_visualization_suggestions(st.session_state.llm_response)
                
                if not viz_suggestions:
                    # 如果LLM没有返回有效建议，使用内置推荐
                    viz_suggestions = get_recommended_visualizations(df)
                
                # 显示可视化建议
                st.subheader("推荐的可视化方式")
                
                for i, viz in enumerate(viz_suggestions):
                    with st.expander(f"{viz.get('name', '未命名')} - {viz.get('reason', '无原因')}"):
                        st.write(f"**推荐原因**: {viz.get('reason', '无')}")
                        
                        # 创建可视化控件
                        viz_type = viz.get('name', '散点图')
                        
                        # 根据可视化类型选择适当的列
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        all_cols = df.columns.tolist()
                        
                        # 默认选择
                        default_x = numeric_cols[0] if numeric_cols else all_cols[0]
                        default_y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)
                        
                        # 根据图表类型配置控件
                        if viz_type in ["散点图", "折线图", "柱状图"]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                x_col = st.selectbox(f"X轴 ({i})", all_cols, key=f"x_{i}")
                            with col2:
                                y_col = st.selectbox(f"Y轴 ({i})", all_cols, key=f"y_{i}")
                            with col3:
                                color_col = st.selectbox(f"颜色 ({i})", [None] + categorical_cols, key=f"color_{i}")
                            
                            fig = create_visualization(df, viz_type, x_col, y_col, color_col)
                        
                        elif viz_type == "直方图":
                            x_col = st.selectbox(f"选择列 ({i})", numeric_cols if numeric_cols else all_cols, key=f"x_{i}")
                            fig = create_visualization(df, viz_type, x_col)
                        
                        elif viz_type == "箱线图":
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox(f"分组列 ({i})", categorical_cols if categorical_cols else all_cols, key=f"x_{i}")
                            with col2:
                                y_col = st.selectbox(f"数值列 ({i})", numeric_cols if numeric_cols else all_cols, key=f"y_{i}")
                            
                            fig = create_visualization(df, viz_type, x_col, y_col)
                        
                        elif viz_type == "饼图":
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox(f"分类列 ({i})", categorical_cols if categorical_cols else all_cols, key=f"x_{i}")
                            with col2:
                                y_col = st.selectbox(f"数值列 (可选) ({i})", [None] + numeric_cols, key=f"y_{i}")
                            
                            fig = create_visualization(df, viz_type, x_col, y_col)
                        
                        elif viz_type in ["热力图", "相关性矩阵"]:
                            fig = create_visualization(df, "相关性矩阵")
                        
                        else:
                            # 默认散点图
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox(f"X轴 ({i})", all_cols, key=f"x_{i}")
                            with col2:
                                y_col = st.selectbox(f"Y轴 ({i})", all_cols, key=f"y_{i}")
                            
                            fig = create_visualization(df, "散点图", x_col, y_col)
                        
                        # 显示图表
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法创建此可视化，请尝试不同的列或图表类型")

# 模型训练与预测页面
elif page == "模型训练与预测":
    st.title("模型训练与预测")
    
    if st.session_state.data is None:
        st.warning("请先上传数据文件")
    else:
        df = st.session_state.data
        
        # 选择目标列
        target_col = st.selectbox("选择目标列", df.columns.tolist())
        st.session_state.target_column = target_col
        
        # 获取模型建议
        st.subheader("推荐的机器学习模型")
        
        # 从LLM获取建议
        model_suggestions = []
        if st.session_state.llm_response:
            model_suggestions = parse_llm_model_suggestions(st.session_state.llm_response)
        
        # 如果LLM没有返回有效建议，使用内置推荐
        if not model_suggestions:
            model_suggestions = get_model_suggestions(df, target_col)
        
        # 显示模型建议
        for i, model in enumerate(model_suggestions):
            with st.expander(f"{model.get('name', '未命名')} - {model.get('reason', '无原因')}"):
                st.write(f"**推荐原因**: {model.get('reason', '无')}")
                st.write(f"**优点**: {model.get('pros', '无')}")
                st.write(f"**缺点**: {model.get('cons', '无')}")
                st.write(f"**适用场景**: {model.get('use_case', '无')}")
                
                # 训练模型按钮
                if st.button(f"训练 {model.get('name', '模型')}", key=f"train_{i}"):
                    with st.spinner(f"正在训练 {model.get('name', '模型')}..."):
                        model_name = model.get('name', '未知模型')
                        trained_model, metrics, error = train_model(df, target_col, model_name)
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state.trained_model = trained_model
                            st.session_state.model_metrics = metrics
                            st.success(f"{model_name} 训练完成!")
        
        # 显示训练结果
        if st.session_state.trained_model and st.session_state.model_metrics:
            st.subheader("模型训练结果")
            
            # 显示评估指标
            st.write("**评估指标**")
            metrics_df = pd.DataFrame(st.session_state.model_metrics, index=[0])
            st.dataframe(metrics_df)
            
            # 预测部分
            st.subheader("使用模型进行预测")
            
            # 创建输入表单
            with st.form("prediction_form"):
                # 为每个特征创建输入字段
                feature_inputs = {}
                feature_cols = [col for col in df.columns if col != target_col]
                
                # 根据特征类型创建不同的输入控件
                for col in feature_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # 数值型特征
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        mean_val = float(df[col].mean())
                        feature_inputs[col] = st.slider(
                            f"{col}", 
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val-min_val)/100
                        )
                    else:
                        # 分类特征
                        options = df[col].unique().tolist()
                        feature_inputs[col] = st.selectbox(f"{col}", options)
                
                # 提交按钮
                submit_button = st.form_submit_button("预测")
            
            # 处理预测
            if submit_button:
                prediction, error = predict(
                    st.session_state.trained_model, 
                    feature_inputs, 
                    df, 
                    target_col
                )
                
                if error:
                    st.error(error)
                else:
                    st.success(f"预测结果: {prediction}")
                    
                    # 显示预测结果的可视化（如果适用）
                    if pd.api.types.is_numeric_dtype(df[target_col]):
                        # 对于回归任务，显示预测值与实际值的分布
                        fig = px.histogram(df, x=target_col, title=f"{target_col} 分布")
                        fig.add_vline(x=prediction, line_dash="dash", line_color="red", annotation_text="预测值")
                        st.plotly_chart(fig, use_container_width=True)

# 设置页面
elif page == "设置":
    st.title("设置")
    
    st.subheader("大语言模型配置")
    
    # 加载当前配置
    config = load_config()
    
    # 创建表单
    with st.form("llm_config_form"):
        # OpenAI配置
        st.write("**OpenAI配置**")
        openai_config = config["llm_models"].get("openai", {})
        openai_api_key = st.text_input(
            "OpenAI API密钥", 
            value=openai_config.get("api_key", ""),
            type="password"
        )
        openai_model = st.selectbox(
            "OpenAI模型",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0 if not openai_config.get("model") else ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"].index(openai_config.get("model"))
        )
        openai_api_base = st.text_input(
            "API地址 (可选，留空使用默认地址)",
            value=openai_config.get("api_base", "")
        )
        
        # 其他模型配置可以在这里添加
        # ...
        
        # 提交按钮
        submit_button = st.form_submit_button("保存配置")
    
    # 处理表单提交
    if submit_button:
        # 更新OpenAI配置
        update_model_config("openai", openai_api_key, openai_model, openai_api_base)
        
        # 更新其他模型配置
        # ...
        
        st.success("配置已保存")
    
    # 显示当前配置
    with st.expander("查看当前配置"):
        st.json(config)
    
    # 添加使用说明
    st.subheader("使用说明")
    st.markdown("""
    ### 数据上传与分析
    1. 上传CSV或TXT格式的数据文件
    2. 查看数据基本信息和统计数据
    3. 点击"获取AI分析建议"获取数据可视化和模型推荐
    4. 探索推荐的可视化图表
    
    ### 模型训练与预测
    1. 选择目标列（要预测的变量）
    2. 从推荐的模型中选择一个进行训练
    3. 查看模型评估指标
    4. 输入特征值进行预测
    
    ### 设置
    1. 配置大语言模型API密钥和模型
    2. 保存配置以便后续使用
    """)

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("数据分析与机器学习应用 v1.0")