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


# --- Helper function ---
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


# --- Page UI Setup ---
st.set_page_config(page_title="MapCrunch AI Agent", layout="wide")
st.title("🗺️ MapCrunch AI Agent")
st.caption(
    "An AI agent that explores and identifies geographic locations through multi-step interaction."
)

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Agent Configuration")

    # Get API keys from HF Secrets (must be set in Space settings when deploying)
    os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", "")
    os.environ["ANTHROPIC_API_KEY"] = st.secrets.get("ANTHROPIC_API_KEY", "")
    # os.environ['GOOGLE_API_KEY'] = st.secrets.get("GOOGLE_API_KEY", "")

    # Dataset selection
    available_datasets = get_available_datasets()
    dataset_choice = st.selectbox("Select Dataset", available_datasets)

    model_choice = st.selectbox("Select AI Model", list(MODELS_CONFIG.keys()))
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

    config = MODELS_CONFIG.get(model_choice)
    model_class = globals()[config["class"]]
    model_instance_name = config["model_name"]

    # Initialize helpers and result lists
    benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_choice)
    all_results = []

    st.info(
        f"Starting Agent Benchmark... Dataset: {dataset_choice}, Model: {model_choice}, Steps: {steps_per_sample}, Samples: {num_samples_to_run}"
    )

    overall_progress_bar = st.progress(0, text="Overall Progress")

    # Initialize the bot outside the loop to reuse the browser instance for efficiency
    with st.spinner("Initializing browser and AI model..."):
        # Note: Must run in headless mode on HF Spaces
        bot = GeoBot(model=model_class, model_name=model_instance_name, headless=True)

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
            reasoning_placeholder.info(
                f"Thinking... (Step {step_num}/{steps_per_sample})"
            )
            action_placeholder.empty()

            # Observe and label arrows
            bot.controller.label_arrows_on_screen()
            screenshot_bytes = bot.controller.take_street_view_screenshot()
            image_placeholder.image(
                screenshot_bytes, caption=f"Step {step_num} View", use_column_width=True
            )

            # Update history
            history.append(
                {
                    "image_b64": bot.pil_to_base64(
                        Image.open(BytesIO(screenshot_bytes))
                    ),
                    "action": "N/A",
                }
            )

            # Think
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

            # Force a GUESS on the last step
            if step_num == steps_per_sample and action != "GUESS":
                st.warning("Max steps reached. Forcing a GUESS action.")
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
                        "GUESS action was missing coordinates. Guess failed for this sample."
                    )
                break  # End exploration for the current sample

            elif action == "MOVE_FORWARD":
                bot.controller.move("forward")
            elif action == "MOVE_BACKWARD":
                bot.controller.move("backward")
            elif action == "PAN_LEFT":
                bot.controller.pan_view("left")
            elif action == "PAN_RIGHT":
                bot.controller.pan_view("right")

            time.sleep(1)  # A brief pause between steps for better visualization

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
