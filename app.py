import gradio as gr
import json
import os
import time
from io import BytesIO
from PIL import Image

# 导入项目的核心逻辑和配置
from geo_bot import GeoBot, AGENT_PROMPT_TEMPLATE
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, DATA_PATHS
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 全局设置 ---
# 从HF Secrets安全地读取API密钥
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")
# os.environ['GOOGLE_API_KEY'] = os.environ.get("GOOGLE_API_KEY", "")

# 加载golden labels数据
try:
    with open(DATA_PATHS["golden_labels"], "r", encoding="utf-8") as f:
        GOLDEN_LABELS = json.load(f).get("samples", [])
except FileNotFoundError:
    print(f"警告: 数据文件 '{DATA_PATHS['golden_labels']}' 未找到。")
    GOLDEN_LABELS = []


# --- 核心处理函数 (使用yield实现流式更新) ---
def run_agent_process(
    model_choice, steps_per_sample, sample_index, progress=gr.Progress(track_tqdm=True)
):
    """
    这个函数是整个应用的引擎，它是一个生成器 (generator)，会逐步yield更新。
    """
    # 1. 初始化环境
    yield {
        status_text: "状态: 正在初始化浏览器和AI模型...",
        image_output: None,
        reasoning_output: "",
        action_output: "",
        result_output: "",
    }

    config = MODELS_CONFIG.get(model_choice)
    model_class = globals()[config["class"]]
    model_instance_name = config["model_name"]
    bot = GeoBot(model=model_class, model_name=model_instance_name, headless=True)

    # 2. 加载选定的样本位置
    sample = GOLDEN_LABELS[sample_index]
    ground_truth = {"lat": sample.get("lat"), "lng": sample.get("lng")}

    if not bot.controller.load_location_from_data(sample):
        yield {status_text: "错误: 加载地图位置失败。请重试。"}
        return

    bot.controller.setup_clean_environment()

    history = []
    final_guess = None

    # 3. 开始多步探索循环
    for step in range(steps_per_sample):
        step_num = step + 1
        yield {status_text: f"状态: 探索中... (第 {step_num}/{steps_per_sample} 步)"}

        # a. 观察 (Observe)
        bot.controller.label_arrows_on_screen()
        screenshot_bytes = bot.controller.take_street_view_screenshot()

        # b. 思考 (Think)
        current_screenshot_b64 = bot.pil_to_base64(
            Image.open(BytesIO(screenshot_bytes))
        )
        history.append({"image_b64": current_screenshot_b64, "action": "N/A"})

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

        if not decision:
            decision = {
                "action_details": {"action": "PAN_RIGHT"},
                "reasoning": "Default recovery.",
            }

        action = decision.get("action_details", {}).get("action")
        reasoning = decision.get("reasoning", "N/A")
        history[-1]["action"] = action

        # c. 更新UI
        yield {
            image_output: Image.open(BytesIO(screenshot_bytes)),
            reasoning_output: f"**AI Reasoning:**\n\n{reasoning}",
            action_output: f"**AI Action:** `{action}`",
        }

        # d. 强制在最后一步猜测
        if step_num == steps_per_sample and action != "GUESS":
            action = "GUESS"
            yield {status_text: "状态: 已达最大步数，强制执行GUESS..."}

        # e. 行动 (Act)
        if action == "GUESS":
            lat, lon = (
                decision.get("action_details", {}).get("lat"),
                decision.get("action_details", {}).get("lon"),
            )
            if lat is not None and lon is not None:
                final_guess = (lat, lon)
            break
        elif action == "MOVE_FORWARD":
            bot.controller.move("forward")
        elif action == "MOVE_BACKWARD":
            bot.controller.move("backward")
        elif action == "PAN_LEFT":
            bot.controller.pan_view("left")
        elif action == "PAN_RIGHT":
            bot.controller.pan_view("right")

        time.sleep(1)  # 步骤间稍作停顿

    # 4. 循环结束，计算最终结果并更新UI
    yield {status_text: "状态: 探索完成，正在计算最终结果..."}

    if final_guess:
        distance = bot.calculate_distance(ground_truth, final_guess)
        result_text = f"""
        ### 📍 最终结果
        - **真实位置:** `Lat: {ground_truth["lat"]:.4f}, Lon: {ground_truth["lng"]:.4f}`
        - **Agent猜测:** `Lat: {final_guess[0]:.4f}, Lon: {final_guess[1]:.4f}`
        - **距离误差:** `{distance:.1f} km`
        """
        yield {result_output: result_text, status_text: "状态: 完成！"}
    else:
        yield {
            result_output: "### 📍 最终结果\n\nAgent 未能做出有效猜测。",
            status_text: "状态: 完成！",
        }

    bot.close()  # 关闭浏览器


# --- Gradio UI 布局 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗺️ 可视化 GeoBot 智能体")
    gr.Markdown("选择配置并启动Agent，观察它如何通过探索来猜测自己的地理位置。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ 控制面板")
            model_choice = gr.Dropdown(
                list(MODELS_CONFIG.keys()), label="选择AI模型", value="gpt-4o"
            )
            steps_per_sample = gr.Slider(
                3, 20, value=10, step=1, label="每轮最大探索步数"
            )
            sample_index = gr.Dropdown(
                [f"样本 {i}" for i in range(len(GOLDEN_LABELS))],
                label="选择测试样本",
                value="样本 0",
            )
            start_button = gr.Button("🚀 启动智能体", variant="primary")
            status_text = gr.Markdown("状态: 等待启动")
            result_output = gr.Markdown()

        with gr.Column(scale=3):
            gr.Markdown("## 🕵️ Agent探索过程")
            image_output = gr.Image(label="Agent当前视角", height=600)
            with gr.Row():
                reasoning_output = gr.Markdown(label="AI 思考")
                action_output = gr.Markdown(label="AI 行动")

    # 将按钮点击事件连接到核心函数
    # `lambda s: int(s.split(' ')[1])` 用于从"样本 0"中提取出数字0
    start_button.click(
        fn=run_agent_process,
        inputs=[model_choice, steps_per_sample, sample_index],
        outputs=[
            status_text,
            image_output,
            reasoning_output,
            action_output,
            result_output,
        ],
        # `js` 参数用于在点击按钮后禁用它，防止重复点击
        js="""
        (model_choice, steps_per_sample, sample_index) => {
            return [
                "状态: 初始化中...", 
                null, 
                "...", 
                "...", 
                ""
            ];
        }
        """,
    )

if __name__ == "__main__":
    demo.launch()
