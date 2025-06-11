---
title: Omniscient
emoji: 👁️‍🗨️
colorFrom: indigo
colorTo: purple
sdk: streamlit
python_version: 3.11
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

<div align="center">

# 🧠 Omniscient 
### *"The all-knowing AI that sees everything, knows everything"*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow.svg)](https://huggingface.co/spaces/Omniscient001/Omniscient)

*A versatile AI bot for image analysis and dataset curation with support for multiple AI models*

🎮 **[Try it Live on HuggingFace!](https://huggingface.co/spaces/Omniscient001/Omniscient)** *(Actively WIP)*

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🗃️ **Dataset Curation**
Generate and curate high-quality image datasets with intelligent filtering and categorization.

### 🔍 **Single Image Analysis** 
Benchmark different AI models on individual images with detailed performance metrics.

</td>
<td width="50%">

### 🤖 **Agentic Analysis**
Multi-step AI reasoning and analysis with advanced decision-making capabilities.

### 🌐 **Multiple AI Providers**
Seamless integration with OpenAI, Anthropic, and Google AI platforms.

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 📋 **Step 1: Setup Environment**

```bash
cd simple_G_ai_bot
```

Create a `.env` file in the project root:

```bash
# 🔐 .env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 📦 **Step 2: Install Dependencies**

```bash
uv sync
```

### 🎯 **Step 3: Usage Examples**

<details>
<summary><b>🏗️ Dataset Curation</b></summary>

Generate 50 urban outdoor samples:
```bash
python main.py --mode data --samples 50 --urban --no-indoor
```

</details>

<details>
<summary><b>⚡ Single Image Analysis</b></summary>

Benchmark GPT-4o on 5 samples:
```bash
python main.py --mode benchmark --models gpt-4o --samples 5
```

</details>

<details>
<summary><b>🧠 Agentic Analysis</b></summary>

Run multi-step analysis with Gemini:
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 10 --samples 5
```

</details>

---

## ⚙️ Configuration

### 🔑 **Environment Variables**

| Variable | Description | Status |
|:---------|:------------|:------:|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | 🔶 Optional |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models | 🔶 Optional |
| `GOOGLE_API_KEY` | Google AI API key for Gemini models | 🔶 Optional |

### 🛠️ **Command Line Options**

#### 🌟 **Common Options**
- `--mode` → Operation mode (`data`, `benchmark`, `agent`)
- `--samples` → Number of samples to process *(default: 10)*

#### 🏙️ **Data Mode Options**  
- `--urban` → Focus on urban environments
- `--no-indoor` → Exclude indoor scenes

#### 📊 **Benchmark Mode Options**
- `--models` → AI model to use *(e.g., `gpt-4o`, `claude-3`, `gemini-pro`)*

#### 🤖 **Agent Mode Options**
- `--model` → AI model for agentic analysis  
- `--steps` → Number of reasoning steps *(default: 5)*

---

## 🎯 Supported Models

<div align="center">

| Provider | Models | Status |
|:--------:|:-------|:------:|
| **🔵 OpenAI** | GPT-4o, GPT-4, GPT-3.5-turbo | ✅ Active |
| **🟣 Anthropic** | Claude-3-opus, Claude-3-sonnet, Claude-3-haiku | ✅ Active |
| **🔴 Google** | Gemini-2.5-pro, Gemini-pro, Gemini-pro-vision | ✅ Active |

</div>

---

## 📋 Requirements

> **Prerequisites:**
> - 🐍 Python 3.8+
> - 📦 UV package manager  
> - 🔑 Valid API keys for desired AI providers

---

## 🔧 Installation

<table>
<tr>
<td>

**1️⃣** Clone the repository
```bash
git clone <repository-url>
```

**2️⃣** Navigate to project directory
```bash
cd simple_G_ai_bot
```

</td>
<td>

**3️⃣** Create `.env` file with your API keys
```bash
touch .env
# Add your API keys
```

**4️⃣** Install dependencies
```bash
uv sync
```

</td>
</tr>
</table>

**5️⃣** Run the bot with desired mode and options! 🎉

---

## 💡 Examples

### 🏗️ **Basic Dataset Generation**
```bash
python main.py --mode data --samples 20
```

### 🌆 **Urban Scene Analysis**  
```bash
python main.py --mode data --samples 30 --urban --no-indoor
```

### ⚔️ **Model Comparison**
```bash
# GPT-4o Analysis
python main.py --mode benchmark --models gpt-4o --samples 10

# Claude-3 Analysis  
python main.py --mode benchmark --models claude-3-opus --samples 10
```

### 🧠 **Advanced Agentic Workflow**
```bash
python main.py --mode agent --model gemini-2.5-pro --steps 15 --samples 3
```

---

## 🔐 Security Note

> ⚠️ **Important**: Never commit your `.env` file to version control. Add `.env` to your `.gitignore` file to keep your API keys secure.

---

<div align="center">

## 📜 License

**MIT License** - see [LICENSE](LICENSE) file for details.

---

<img src="https://img.shields.io/badge/Made%20with-❤️-red.svg" alt="Made with love">
<img src="https://img.shields.io/badge/AI%20Powered-🤖-blue.svg" alt="AI Powered">
<img src="https://img.shields.io/badge/Open%20Source-💚-green.svg" alt="Open Source">

**⭐ Star this repo if you find it useful!**

</div>