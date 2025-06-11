import streamlit as st
import json
import os
import time
from io import BytesIO
from PIL import Image
from typing import Dict, List, Any

# 导入项目的核心逻辑和配置
from geo_bot import (
    GeoBot,
    AGENT_PROMPT_TEMPLATE,
    BENCHMARK_PROMPT,
)  # 导入Prompt模板以供复用
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, DATA_PATHS, SUCCESS_THRESHOLD_KM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 页面UI设置 ---
st.set_page_config(page_title="MapCrunch AI Agent", layout="wide")
st.title("🗺️ MapCrunch AI Agent")
st.caption("一个通过多步交互探索和识别地理位置的AI智能体")

# --- Sidebar用于配置 ---
with st.sidebar:
    st.header("⚙️ 运行配置")

    # 从HF Secrets获取API密钥 (部署到HF Spaces时，需要在Settings->Secrets中设置)
    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = st.secrets.get("ANTHROPIC_API_KEY", "")
    # 添加其他你可能需要的API密钥
    # os.environ['GOOGLE_API_KEY'] = st.secrets.get("GOOGLE_API_KEY", "")

    model_choice = st.selectbox("选择AI模型", list(MODELS_CONFIG.keys()))
    steps_per_sample = st.slider(
        "每轮最大探索步数", min_value=3, max_value=20, value=10
    )

    # 加载golden labels以供选择
    try:
        with open(DATA_PATHS["golden_labels"], "r", encoding="utf-8") as f:
            golden_labels = json.load(f).get("samples", [])
        total_samples = len(golden_labels)
        num_samples_to_run = st.slider(
            "选择测试样本数量", min_value=1, max_value=total_samples, value=3
        )
    except FileNotFoundError:
        st.error(f"数据文件 '{DATA_PATHS['golden_labels']}' 未找到。请先准备数据。")
        golden_labels = []
        num_samples_to_run = 0

    start_button = st.button(
        "🚀 启动Agent Benchmark", disabled=(num_samples_to_run == 0), type="primary"
    )

# --- Agent运行逻辑 ---
if start_button:
    # 准备运行环境
    test_samples = golden_labels[:num_samples_to_run]

    config = MODELS_CONFIG.get(model_choice)
    model_class = globals()[config["class"]]
    model_instance_name = config["model_name"]

    # 初始化用于统计结果的辅助类和列表
    benchmark_helper = MapGuesserBenchmark()
    all_results = []

    st.info(
        f"即将开始Agent Benchmark... 模型: {model_choice}, 步数: {steps_per_sample}, 样本数: {num_samples_to_run}"
    )

    # 创建一个总进度条
    overall_progress_bar = st.progress(0, text="总进度")

    # 初始化Bot (注意：在HF Spaces上，必须以headless模式运行)
    # 将Bot的初始化放在循环外，可以复用同一个浏览器实例，提高效率
    with st.spinner("正在初始化浏览器和AI模型..."):
        bot = GeoBot(model=model_class, model_name=model_instance_name, headless=True)

    # 主循环，遍历所有选择的测试样本
    for i, sample in enumerate(test_samples):
        sample_id = sample.get("id", "N/A")
        st.divider()
        st.header(f"▶️ 运行中:样本 {i + 1}/{num_samples_to_run} (ID: {sample_id})")

        # 加载地图位置
        if not bot.controller.load_location_from_data(sample):
            st.error(f"加载样本 {sample_id} 失败，已跳过。")
            continue

        bot.controller.setup_clean_environment()

        # 为当前样本创建可视化布局
        col1, col2 = st.columns([2, 3])
        with col1:
            image_placeholder = st.empty()
        with col2:
            reasoning_placeholder = st.empty()
            action_placeholder = st.empty()

        # --- 内部的Agent探索循环 ---
        history = []
        final_guess = None

        for step in range(steps_per_sample):
            step_num = step + 1
            reasoning_placeholder.info(
                f"思考中... (第 {step_num}/{steps_per_sample} 步)"
            )
            action_placeholder.empty()

            # 观察并标记箭头
            bot.controller.label_arrows_on_screen()
            screenshot_bytes = bot.controller.take_street_view_screenshot()
            image_placeholder.image(
                screenshot_bytes, caption=f"Step {step_num} View", use_column_width=True
            )

            # 更新历史
            history.append(
                {
                    "image_b64": bot.pil_to_base64(
                        Image.open(BytesIO(screenshot_bytes))
                    ),
                    "action": "N/A",
                }
            )

            # 思考
            prompt = AGENT_PROMPT_TEMPLATE.format(
                remaining_steps=steps_per_sample - step,
                history_text="\n".join(
                    [f"Step {j + 1}: {h['action']}" for j, h in enumerate(history)]
                ),
                available_actions=json.dumps(bot.controller.get_available_actions()),
            )
            message = bot._create_message_with_history(
                prompt, [h["image_b64"] for h in history]
            )
            response = bot.model.invoke(message)
            decision = bot._parse_agent_response(response)

            if not decision:  # Fallback
                decision = {
                    "action_details": {"action": "PAN_RIGHT"},
                    "reasoning": "Default recovery.",
                }

            action = decision.get("action_details", {}).get("action")
            history[-1]["action"] = action

            reasoning_placeholder.info(
                f"**AI Reasoning:**\n\n{decision.get('reasoning', 'N/A')}"
            )
            action_placeholder.success(f"**AI Action:** `{action}`")

            # 强制在最后一步进行GUESS
            if step_num == steps_per_sample and action != "GUESS":
                st.warning("已达最大步数，强制执行GUESS。")
                action = "GUESS"

            # 行动
            if action == "GUESS":
                lat, lon = (
                    decision.get("action_details", {}).get("lat"),
                    decision.get("action_details", {}).get("lon"),
                )
                if lat is not None and lon is not None:
                    final_guess = (lat, lon)
                else:
                    # 如果AI没在GUESS时提供坐标，再问一次
                    # (这里的简化处理是直接结束，但在更复杂的版本可以再调用一次AI)
                    st.error("GUESS动作中缺少坐标，本次猜测失败。")
                break  # 结束当前样本的探索

            elif action == "MOVE_FORWARD":
                bot.controller.move("forward")
            elif action == "MOVE_BACKWARD":
                bot.controller.move("backward")
            elif action == "PAN_LEFT":
                bot.controller.pan_view("left")
            elif action == "PAN_RIGHT":
                bot.controller.pan_view("right")

            time.sleep(1)  # 在步骤之间稍作停顿，改善视觉效果

        # --- 单个样本运行结束，计算并展示结果 ---
        true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
        distance_km = None
        is_success = False

        if final_guess:
            distance_km = benchmark_helper.calculate_distance(true_coords, final_guess)
            if distance_km is not None:
                is_success = distance_km <= SUCCESS_THRESHOLD_KM

            st.subheader("🎯 本轮结果")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                "最终猜测 (Lat, Lon)", f"{final_guess[0]:.3f}, {final_guess[1]:.3f}"
            )
            res_col2.metric(
                "真实位置 (Lat, Lon)",
                f"{true_coords['lat']:.3f}, {true_coords['lng']:.3f}",
            )
            res_col3.metric(
                "距离误差",
                f"{distance_km:.1f} km" if distance_km is not None else "N/A",
                delta=f"{'成功' if is_success else '失败'}",
                delta_color=("inverse" if is_success else "off"),
            )
        else:
            st.error("Agent 未能做出最终猜测。")

        all_results.append(
            {
                "sample_id": sample_id,
                "model": model_choice,
                "true_coordinates": true_coords,
                "predicted_coordinates": final_guess,
                "distance_km": distance_km,
                "success": is_success,
            }
        )

        # 更新总进度条
        overall_progress_bar.progress(
            (i + 1) / num_samples_to_run, text=f"总进度: {i + 1}/{num_samples_to_run}"
        )

    # --- 所有样本运行完毕，显示最终摘要 ---
    bot.close()  # 关闭浏览器
    st.divider()
    st.header("🏁 Benchmark 最终摘要")

    summary = benchmark_helper.generate_summary(all_results)
    if summary and model_choice in summary:
        stats = summary[model_choice]
        sum_col1, sum_col2 = st.columns(2)
        sum_col1.metric("总成功率", f"{stats.get('success_rate', 0) * 100:.1f} %")
        sum_col2.metric("平均距离误差", f"{stats.get('average_distance_km', 0):.1f} km")
        st.dataframe(all_results)  # 显示详细结果表格
    else:
        st.warning("没有足够的结果来生成摘要。")
