import streamlit as st
import json
import os
import time
from io import BytesIO
from PIL import Image
from pathlib import Path

from geo_bot import GeoBot, AGENT_PROMPT_TEMPLATE
from benchmark import MapGuesserBenchmark
from config import MODELS_CONFIG, get_data_paths, SUCCESS_THRESHOLD_KM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from hf_chat import HuggingFaceChat

# Simple API key setup
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]


def get_available_datasets():
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return ["default"]
    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            data_paths = get_data_paths(dataset_dir.name)
            if os.path.exists(data_paths["golden_labels"]):
                datasets.append(dataset_dir.name)
    return datasets if datasets else ["default"]


def get_model_class(class_name):
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


# UI Setup
st.set_page_config(page_title="MapCrunch AI Agent", layout="wide")
st.title("🗺️ MapCrunch AI Agent")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    dataset_choice = st.selectbox("Dataset", get_available_datasets())
    model_choice = st.selectbox("Model", list(MODELS_CONFIG.keys()))
    steps_per_sample = st.slider("Max Steps", 3, 20, 10)

    # Load dataset
    data_paths = get_data_paths(dataset_choice)
    with open(data_paths["golden_labels"], "r") as f:
        golden_labels = json.load(f).get("samples", [])

    st.info(f"Dataset has {len(golden_labels)} samples")
    num_samples = st.slider(
        "Samples to Test", 1, len(golden_labels), min(3, len(golden_labels))
    )

    start_button = st.button("🚀 Start", type="primary")

# Main Logic
if start_button:
    test_samples = golden_labels[:num_samples]
    config = MODELS_CONFIG[model_choice]
    model_class = get_model_class(config["class"])

    benchmark_helper = MapGuesserBenchmark(dataset_name=dataset_choice)
    all_results = []

    progress_bar = st.progress(0)

    with GeoBot(
        model=model_class, model_name=config["model_name"], headless=True
    ) as bot:
        for i, sample in enumerate(test_samples):
            st.divider()
            st.header(f"Sample {i + 1}/{num_samples} - ID: {sample.get('id', 'N/A')}")

            bot.controller.load_location_from_data(sample)
            bot.controller.setup_clean_environment()

            # Create scrollable container for this sample
            sample_container = st.container()

            with sample_container:
                # Initialize step tracking
                history = []
                final_guess = None

                for step in range(steps_per_sample):
                    step_num = step + 1

                    # Create step container
                    with st.container():
                        st.subheader(f"Step {step_num}/{steps_per_sample}")

                        # Take screenshot and show
                        bot.controller.label_arrows_on_screen()
                        screenshot_bytes = bot.controller.take_street_view_screenshot()

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.image(
                                screenshot_bytes,
                                caption=f"What AI sees",
                                use_column_width=True,
                            )

                        with col2:
                            # Build history for AI
                            current_step = {
                                "image_b64": bot.pil_to_base64(
                                    Image.open(BytesIO(screenshot_bytes))
                                ),
                                "action": "N/A",
                            }
                            history.append(current_step)

                            available_actions = bot.controller.get_available_actions()
                            history_text = "\n".join(
                                [
                                    f"Step {j + 1}: {h['action']}"
                                    for j, h in enumerate(history[:-1])
                                ]
                            )
                            if not history_text:
                                history_text = "First step."

                            prompt = AGENT_PROMPT_TEMPLATE.format(
                                remaining_steps=steps_per_sample - step,
                                history_text=history_text,
                                available_actions=json.dumps(available_actions),
                            )

                            # Show AI context
                            st.write("**Available Actions:**")
                            st.code(json.dumps(available_actions, indent=2))

                            st.write("**AI Context:**")
                            st.text_area(
                                "History",
                                history_text,
                                height=100,
                                disabled=True,
                                key=f"history_{i}_{step}",
                            )

                            # Force guess on last step or get AI decision
                            if step_num == steps_per_sample:
                                action = "GUESS"
                                st.warning("Max steps reached. Forcing GUESS.")
                            else:
                                # Get AI response
                                message = bot._create_message_with_history(
                                    prompt, [h["image_b64"] for h in history]
                                )
                                response = bot.model.invoke(message)
                                decision = bot._parse_agent_response(response)

                                if decision is None:
                                    raise ValueError(
                                        f"Failed to parse AI response: {response.content}"
                                    )

                                action = decision["action_details"]["action"]
                                history[-1]["action"] = action

                                # Show AI decision
                                st.write("**AI Reasoning:**")
                                st.info(decision.get("reasoning", "N/A"))

                                st.write("**AI Action:**")
                                st.success(f"`{action}`")

                                # Show raw response
                                with st.expander("Raw AI Response"):
                                    st.text(response.content)

                        # Execute action
                        if action == "GUESS":
                            if step_num == steps_per_sample:
                                # Forced guess - use fallback coordinates
                                lat, lon = 0.0, 0.0
                                st.error("Forced guess with fallback coordinates")
                            else:
                                lat = decision.get("action_details", {}).get("lat")
                                lon = decision.get("action_details", {}).get("lon")

                            if lat is not None and lon is not None:
                                final_guess = (lat, lon)
                                st.success(f"Final Guess: {lat:.4f}, {lon:.4f}")
                            break
                        elif action == "MOVE_FORWARD":
                            bot.controller.move("forward")
                        elif action == "MOVE_BACKWARD":
                            bot.controller.move("backward")
                        elif action == "PAN_LEFT":
                            bot.controller.pan_view("left")
                        elif action == "PAN_RIGHT":
                            bot.controller.pan_view("right")

                        # Auto scroll to bottom
                        st.empty()  # Force refresh to show latest content
                        time.sleep(1)

                # Sample Results
                st.subheader("Sample Result")
                true_coords = {"lat": sample.get("lat"), "lng": sample.get("lng")}
                distance_km = None
                is_success = False

                if final_guess:
                    distance_km = benchmark_helper.calculate_distance(
                        true_coords, final_guess
                    )
                    if distance_km is not None:
                        is_success = distance_km <= SUCCESS_THRESHOLD_KM

                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Final Guess", f"{final_guess[0]:.3f}, {final_guess[1]:.3f}"
                    )
                    col2.metric(
                        "Ground Truth",
                        f"{true_coords['lat']:.3f}, {true_coords['lng']:.3f}",
                    )
                    col3.metric(
                        "Distance",
                        f"{distance_km:.1f} km",
                        delta="Success" if is_success else "Failed",
                    )
                else:
                    st.error("No final guess made")

                all_results.append(
                    {
                        "sample_id": sample.get("id"),
                        "model": model_choice,
                        "true_coordinates": true_coords,
                        "predicted_coordinates": final_guess,
                        "distance_km": distance_km,
                        "success": is_success,
                    }
                )

            progress_bar.progress((i + 1) / num_samples)

    # Final Summary
    st.divider()
    st.header("🏁 Final Results")

    summary = benchmark_helper.generate_summary(all_results)
    if summary and model_choice in summary:
        stats = summary[model_choice]

        # Overall metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Success Rate", f"{stats.get('success_rate', 0) * 100:.1f}%")
        col2.metric("Average Distance", f"{stats.get('average_distance_km', 0):.1f} km")
        col3.metric("Total Samples", len(all_results))

        # Detailed results table
        st.subheader("Detailed Results")
        st.dataframe(all_results, use_container_width=True)

        # Success breakdown
        successes = [r for r in all_results if r["success"]]
        failures = [r for r in all_results if not r["success"]]

        if successes:
            st.subheader("Successful Samples")
            st.dataframe(successes, use_container_width=True)

        if failures:
            st.subheader("Failed Samples")
            st.dataframe(failures, use_container_width=True)
    else:
        st.error("Could not generate summary")
        st.dataframe(all_results, use_container_width=True)
