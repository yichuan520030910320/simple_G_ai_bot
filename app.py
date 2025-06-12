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
st.set_page_config(page_title="🧠 Omniscient - AI Geographic Analysis", layout="wide")
st.title("🧠 Omniscient")
st.markdown("### *The all-knowing AI that sees everything, knows everything*")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    # Get available datasets and ensure we have a valid default
    available_datasets = get_available_datasets()
    default_dataset = available_datasets[0] if available_datasets else "default"

    dataset_choice = st.selectbox("Dataset", available_datasets, index=0)
    model_choice = st.selectbox("Model", list(MODELS_CONFIG.keys()))
    steps_per_sample = st.slider("Max Steps", 3, 20, 10)
    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.0,
        0.1,
        help="Controls randomness in AI responses. 0.0 = deterministic, higher = more creative",
    )

    # Load dataset with error handling
    data_paths = get_data_paths(dataset_choice)
    try:
        with open(data_paths["golden_labels"], "r") as f:
            golden_labels = json.load(f).get("samples", [])

        st.info(f"Dataset '{dataset_choice}' has {len(golden_labels)} samples")
        if len(golden_labels) == 0:
            st.error(f"Dataset '{dataset_choice}' contains no samples!")
            st.stop()

    except FileNotFoundError:
        st.error(
            f"❌ Dataset '{dataset_choice}' not found at {data_paths['golden_labels']}"
        )
        st.info("💡 Available datasets: " + ", ".join(available_datasets))
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset '{dataset_choice}': {str(e)}")
        st.stop()

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
        model=model_class,
        model_name=config["model_name"],
        headless=True,
        temperature=temperature,
    ) as bot:
        for i, sample in enumerate(test_samples):
            st.divider()
            st.header(f"Sample {i + 1}/{num_samples} - ID: {sample.get('id', 'N/A')}")

            # Load the sample location
            bot.controller.load_location_from_data(sample)

            # Create containers for UI updates
            sample_container = st.container()

            # Initialize UI state for this sample
            step_containers = {}
            sample_steps_data = []

            def ui_step_callback(step_info):
                """Callback function to update UI after each step"""
                step_num = step_info["step_num"]

                # Store step data
                sample_steps_data.append(step_info)

                with sample_container:
                    # Create step container if it doesn't exist
                    if step_num not in step_containers:
                        step_containers[step_num] = st.container()

                    with step_containers[step_num]:
                        st.subheader(f"Step {step_num}/{step_info['max_steps']}")

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            # Display screenshot
                            st.image(
                                step_info["screenshot_bytes"],
                                caption=f"What AI sees - Step {step_num}",
                                use_column_width=True,
                            )

                        with col2:
                            # Show available actions
                            st.write("**Available Actions:**")
                            st.code(
                                json.dumps(step_info["available_actions"], indent=2)
                            )

                            # Show history context - use the history from step_info
                            current_history = step_info.get("history", [])
                            history_text = bot.generate_history_text(current_history)
                            st.write("**AI Context:**")
                            st.text_area(
                                "History",
                                history_text,
                                height=100,
                                disabled=True,
                                key=f"history_{i}_{step_num}",
                            )

                            # Show AI reasoning and action
                            action = step_info.get("action_details", {}).get(
                                "action", "N/A"
                            )

                            if step_info.get("is_final_step") and action != "GUESS":
                                st.warning("Max steps reached. Forcing GUESS.")

                            st.write("**AI Reasoning:**")
                            st.info(step_info.get("reasoning", "N/A"))

                            st.write("**AI Action:**")
                            if action == "GUESS":
                                lat = step_info.get("action_details", {}).get("lat")
                                lon = step_info.get("action_details", {}).get("lon")
                                st.success(f"`{action}` - {lat:.4f}, {lon:.4f}")
                            else:
                                st.success(f"`{action}`")

                            # Show decision details for debugging
                            with st.expander("Decision Details"):
                                decision_data = {
                                    "reasoning": step_info.get("reasoning"),
                                    "action_details": step_info.get("action_details"),
                                    "remaining_steps": step_info.get("remaining_steps"),
                                }
                                st.json(decision_data)

                # Force UI refresh
                time.sleep(0.5)  # Small delay to ensure UI updates are visible

            # Run the agent loop with UI callback
            try:
                final_guess = bot.run_agent_loop(
                    max_steps=steps_per_sample, step_callback=ui_step_callback
                )
            except Exception as e:
                st.error(f"Error during agent execution: {e}")
                final_guess = None

            # Sample Results
            with sample_container:
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
                        "steps_taken": len(sample_steps_data),
                        "max_steps": steps_per_sample,
                        "temperature": temperature,
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

    # Calculate summary stats
    successes = [r for r in all_results if r["success"]]
    success_rate = len(successes) / len(all_results) if all_results else 0

    valid_distances = [
        r["distance_km"] for r in all_results if r["distance_km"] is not None
    ]
    avg_distance = sum(valid_distances) / len(valid_distances) if valid_distances else 0

    # Overall metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Success Rate", f"{success_rate * 100:.1f}%")
    col2.metric("Average Distance", f"{avg_distance:.1f} km")
    col3.metric("Total Samples", len(all_results))

    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(all_results, use_container_width=True)

    # Success/failure breakdown
    if successes:
        st.subheader("✅ Successful Samples")
        st.dataframe(successes, use_container_width=True)

    failures = [r for r in all_results if not r["success"]]
    if failures:
        st.subheader("❌ Failed Samples")
        st.dataframe(failures, use_container_width=True)

    # Export functionality
    if st.button("💾 Export Results"):
        results_json = json.dumps(all_results, indent=2)
        st.download_button(
            label="Download results.json",
            data=results_json,
            file_name=f"geo_results_{dataset_choice}_{model_choice}_{num_samples}samples.json",
            mime="application/json",
        )
