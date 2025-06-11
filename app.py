import streamlit as st
import json
import os
import time
from io import BytesIO
from PIL import Image
from typing import Dict, List, Any
from pathlib import Path

# Import core logic and configurations from the project
from geo_bot import (
    GeoBot,
    AGENT_PROMPT_TEMPLATE,
    BENCHMARK_PROMPT,
)
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, get_data_paths, SUCCESS_THRESHOLD_KM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from hf_chat import HuggingFaceChat


def setup_api_keys():
    """Setup API keys from Streamlit secrets and show status"""
    key_status = {}

    # OpenAI
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        key_status["OpenAI"] = "✅ Available"
    else:
        key_status["OpenAI"] = "❌ Missing"

    # Anthropic
    anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        key_status["Anthropic"] = "✅ Available"
    else:
        key_status["Anthropic"] = "❌ Missing"

    # Google
    google_key = st.secrets.get("GOOGLE_API_KEY", "")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        key_status["Google"] = "✅ Available"
    else:
        key_status["Google"] = "❌ Missing"

    # HuggingFace
    hf_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
    if hf_key:
        os.environ["HUGGINGFACE_API_KEY"] = hf_key
        key_status["HuggingFace"] = "✅ Available"
    else:
        key_status["HuggingFace"] = "❌ Missing"

    return key_status


def get_available_models(key_status):
    """Get available models based on API key status"""
    available_models = {}

    for model_id, config in MODELS_CONFIG.items():
        api_key_env = config["api_key_env"]

        # Check if required API key is available
        if (
            api_key_env == "OPENAI_API_KEY"
            and "OpenAI" in key_status
            and "✅" in key_status["OpenAI"]
        ):
            available_models[model_id] = config
        elif (
            api_key_env == "ANTHROPIC_API_KEY"
            and "Anthropic" in key_status
            and "✅" in key_status["Anthropic"]
        ):
            available_models[model_id] = config
        elif (
            api_key_env == "GOOGLE_API_KEY"
            and "Google" in key_status
            and "✅" in key_status["Google"]
        ):
            available_models[model_id] = config
        elif (
            api_key_env == "HUGGINGFACE_API_KEY"
            and "HuggingFace" in key_status
            and "✅" in key_status["HuggingFace"]
        ):
            if HuggingFaceChat is not None:  # Only if wrapper is available
                available_models[model_id] = config

    return available_models


def get_available_datasets():
    """Get list of available datasets"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return ["default"]

    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            data_paths = get_data_paths(dataset_name)
            if os.path.exists(data_paths["golden_labels"]):
                datasets.append(dataset_name)

    return datasets if datasets else ["default"]


def get_model_class(model_config):
    """Get the appropriate model class based on config"""
    class_name = model_config["class"]
    if class_name == "ChatOpenAI":
        return ChatOpenAI
    elif class_name == "ChatAnthropic":
        return ChatAnthropic
    elif class_name == "ChatGoogleGenerativeAI":
        return ChatGoogleGenerativeAI
    elif class_name == "HuggingFaceChat":
        return HuggingFaceChat
    else:
        raise ValueError(f"Unknown model class: {class_name}")


# --- Page UI Setup ---
st.set_page_config(page_title="MapCrunch AI Agent", layout="wide")
st.title("🗺️ MapCrunch AI Agent")
st.caption(
    "An AI agent that explores and identifies geographic locations through multi-step interaction."
)

# Setup API keys and check status
key_status = setup_api_keys()
available_models = get_available_models(key_status)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Agent Configuration")

    # Show API key status
    with st.expander("🔑 API Key Status", expanded=False):
        for provider, status in key_status.items():
            st.text(f"{provider}: {status}")

        if not any("✅" in status for status in key_status.values()):
            st.error(
                "⚠️ No API keys configured! Please set up API keys in HF Spaces secrets."
            )
            st.info(
                "Add these secrets in your Space settings:\n- OPENAI_API_KEY\n- ANTHROPIC_API_KEY\n- GOOGLE_API_KEY\n- HUGGINGFACE_API_KEY"
            )

    # Dataset selection
    available_datasets = get_available_datasets()
    dataset_choice = st.selectbox("Select Dataset", available_datasets)

    # Model selection (only show available models)
    if not available_models:
        st.error("❌ No models available! Please configure API keys.")
        st.stop()

    model_options = {
        model_id: f"{model_id} - {config['description']}"
        for model_id, config in available_models.items()
    }
    model_choice = st.selectbox(
        "Select AI Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x],
    )

    steps_per_sample = st.slider(
        "Max Exploration Steps per Sample", min_value=3, max_value=20, value=10
    )

    # Load golden labels for selected dataset
    data_paths = get_data_paths(dataset_choice)
    try:
        with open(data_paths["golden_labels"], "r", encoding="utf-8") as f:
            golden_labels = json.load(f).get("samples", [])
        total_samples = len(golden_labels)

        st.info(f"Dataset '{dataset_choice}' has {total_samples} samples")

        num_samples_to_run = st.slider(
            "Number of Samples to Test",
            min_value=1,
            max_value=total_samples,
            value=min(3, total_samples),
        )
    except FileNotFoundError:
        st.error(
            f"Dataset '{dataset_choice}' not found at {data_paths['golden_labels']}. Please create the dataset first."
        )
        golden_labels = []
        num_samples_to_run = 0

    start_button = st.button(
        "🚀 Start Agent Benchmark", disabled=(num_samples_to_run == 0), type="primary"
    )

# --- Agent Execution Logic ---
if start_button:
    # Prepare the environment
    test_samples = golden_labels[:num_samples_to_run]

    config = available_models.get(model_choice)
    if not config:
        st.error(f"Model {model_choice} is not available!")
        st.stop()

    try:
        model_class = get_model_class(config)
        model_instance_name = config["model_name"]
    except Exception as e:
        st.error(f"Failed to load model class: {e}")
        st.stop()

    # Initialize helpers and result lists
    benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_choice)
    all_results = []

    st.info(
        f"Starting Agent Benchmark... Dataset: {dataset_choice}, Model: {model_choice}, Steps: {steps_per_sample}, Samples: {num_samples_to_run}"
    )

    overall_progress_bar = st.progress(0, text="Overall Progress")

    # Initialize the bot outside the loop to reuse the browser instance for efficiency
    with st.spinner("Initializing browser and AI model..."):
        try:
            # Note: Must run in headless mode on HF Spaces
            bot = GeoBot(
                model=model_class, model_name=model_instance_name, headless=True
            )
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")
            st.info("This might be due to API key issues or model unavailability.")
            st.stop()

    # Main loop to iterate through all selected test samples
    for i, sample in enumerate(test_samples):
        sample_id = sample.get("id", "N/A")
        st.divider()
        st.header(f"▶️ Running Sample {i + 1}/{num_samples_to_run} (ID: {sample_id})")

        if not bot.controller.load_location_from_data(sample):
            st.error(f"Failed to load location for sample {sample_id}. Skipping.")
            continue

        bot.controller.setup_clean_environment()

        # Create the visualization layout for the current sample
        col1, col2 = st.columns([2, 3])
        with col1:
            image_placeholder = st.empty()
        with col2:
            reasoning_placeholder = st.empty()
            action_placeholder = st.empty()

        # --- Inner agent exploration loop ---
        history = []
        final_guess = None

        for step in range(steps_per_sample):
            step_num = step + 1
            unique_step_id = f"sample_{i}_step_{step_num}"  # Unique identifier

            reasoning_placeholder.info(
                f"🤔 Thinking... (Step {step_num}/{steps_per_sample})"
            )
            action_placeholder.empty()

            try:
                # Observe and label arrows
                bot.controller.label_arrows_on_screen()
                screenshot_bytes = bot.controller.take_street_view_screenshot()

                # Current view
                image_placeholder.image(
                    screenshot_bytes,
                    caption=f"🔍 Step {step_num} - What AI Sees Now",
                    use_column_width=True,
                )

                # Update history
                current_step_data = {
                    "image_b64": bot.pil_to_base64(
                        Image.open(BytesIO(screenshot_bytes))
                    ),
                    "action": "N/A",
                    "screenshot_bytes": screenshot_bytes,
                    "step_num": step_num,
                }
                history.append(current_step_data)

                # Think
                available_actions = bot.controller.get_available_actions()
                history_text = "\n".join(
                    [f"Step {j + 1}: {h['action']}" for j, h in enumerate(history[:-1])]
                )
                if not history_text:
                    history_text = "No history yet. This is the first step."

                prompt = AGENT_PROMPT_TEMPLATE.format(
                    remaining_steps=steps_per_sample - step,
                    history_text=history_text,
                    available_actions=json.dumps(available_actions),
                )

                # Show what AI is considering
                with reasoning_placeholder:
                    st.info("🧠 **AI is analyzing the situation...**")
                    with st.expander("🔍 Available Actions", expanded=False):
                        st.json(available_actions)

                    # Only show context if there's meaningful history
                    if len(history) > 1:
                        with st.expander("📝 Previous Steps", expanded=False):
                            for j, h in enumerate(history[:-1]):
                                st.write(f"Step {j + 1}: {h.get('action', 'N/A')}")

                message = bot._create_message_with_history(
                    prompt, [h["image_b64"] for h in history]
                )

                # Get AI response
                response = bot.model.invoke(message)
                decision = bot._parse_agent_response(response)

                if not decision:  # Fallback
                    decision = {
                        "action_details": {"action": "PAN_RIGHT"},
                        "reasoning": "⚠️ Response parsing failed. Using default recovery action.",
                    }

                action = decision.get("action_details", {}).get("action")
                history[-1]["action"] = action
                history[-1]["reasoning"] = decision.get("reasoning", "N/A")

                # Display AI's decision
                reasoning_placeholder.success("✅ **AI Decision Made!**")

                with action_placeholder:
                    st.success(f"🎯 **AI Action:** `{action}`")

                    # Show reasoning in expandable section
                    with st.expander("🧠 AI's Reasoning", expanded=True):
                        st.info(decision.get("reasoning", "N/A"))

                        if action == "GUESS":
                            lat = decision.get("action_details", {}).get("lat")
                            lon = decision.get("action_details", {}).get("lon")
                            if lat and lon:
                                st.success(f"📍 **Final Guess:** {lat:.4f}, {lon:.4f}")

                # Force a GUESS on the last step
                if step_num == steps_per_sample and action != "GUESS":
                    st.warning("⏰ Max steps reached. Forcing a GUESS action.")
                    action = "GUESS"

                # Act
                if action == "GUESS":
                    lat, lon = (
                        decision.get("action_details", {}).get("lat"),
                        decision.get("action_details", {}).get("lon"),
                    )
                    if lat is not None and lon is not None:
                        final_guess = (lat, lon)
                    else:
                        st.error(
                            "❌ GUESS action was missing coordinates. Guess failed for this sample."
                        )
                    break  # End exploration for the current sample

                elif action == "MOVE_FORWARD":
                    with st.spinner("🚶 Moving forward..."):
                        bot.controller.move("forward")
                elif action == "MOVE_BACKWARD":
                    with st.spinner("🔄 Moving backward..."):
                        bot.controller.move("backward")
                elif action == "PAN_LEFT":
                    with st.spinner("⬅️ Panning left..."):
                        bot.controller.pan_view("left")
                elif action == "PAN_RIGHT":
                    with st.spinner("➡️ Panning right..."):
                        bot.controller.pan_view("right")

                time.sleep(1)  # A brief pause between steps

            except Exception as e:
                st.error(f"Error in step {step_num}: {e}")
                break

        # --- End of single sample run, calculate and display results ---
        true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
        distance_km = None
        is_success = False

        if final_guess:
            distance_km = benchmark_helper.calculate_distance(true_coords, final_guess)
            if distance_km is not None:
                is_success = distance_km <= SUCCESS_THRESHOLD_KM

            st.subheader("🎯 Round Result")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(
                "Final Guess (Lat, Lon)", f"{final_guess[0]:.3f}, {final_guess[1]:.3f}"
            )
            res_col2.metric(
                "Ground Truth (Lat, Lon)",
                f"{true_coords['lat']:.3f}, {true_coords['lng']:.3f}",
            )
            res_col3.metric(
                "Distance Error",
                f"{distance_km:.1f} km" if distance_km is not None else "N/A",
                delta=f"{'Success' if is_success else 'Failure'}",
                delta_color=("inverse" if is_success else "off"),
            )
        else:
            st.error("Agent failed to make a final guess.")

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

        # Update overall progress bar
        overall_progress_bar.progress(
            (i + 1) / num_samples_to_run,
            text=f"Overall Progress: {i + 1}/{num_samples_to_run}",
        )

    # --- End of all samples, display final summary ---
    bot.close()  # Close the browser
    st.divider()
    st.header("🏁 Benchmark Summary")

    summary = benchmark_helper.generate_summary(all_results)
    if summary and model_choice in summary:
        stats = summary[model_choice]
        sum_col1, sum_col2 = st.columns(2)
        sum_col1.metric(
            "Overall Success Rate", f"{stats.get('success_rate', 0) * 100:.1f} %"
        )
        sum_col2.metric(
            "Average Distance Error", f"{stats.get('average_distance_km', 0):.1f} km"
        )
        st.dataframe(all_results)  # Display the detailed results table
    else:
        st.warning("Not enough results to generate a summary.")
