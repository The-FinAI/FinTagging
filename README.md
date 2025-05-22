<!-- Title -->
<h1 align="center">✨ FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information ✨</h1>

<p align="center">
  📁 <a href="https://huggingface.co/datasets/TheFinAI/FinTagging_Original">Benchmark Data</a> | 📖 <a href="#">Arxiv</a> | 🛠️ <a href="https://github.com/Yan2266336/FinBen">Evaluation Framework</a>
</p>

---

## 🌟 Overview

### 📚 Datasets Released

| 📂 Dataset | 📝 Description |
|------------|----------------|
| [**FinNI-eval**](https://huggingface.co/datasets/TheFinAI/FinNI-eval) | Evaluation set for FinNI subtask within FinTagging benchmark. |
| [**FinCL-eval**](https://huggingface.co/datasets/TheFinAI/FinCL-eval) | Evaluation set for FinCL subtask within FinTagging benchmark. |
| [**FinTagging_Original**](https://huggingface.co/datasets/TheFinAI/FinTagging_Original) | Original benchmark dataset without preprocessing, suitable for custom research. Annotated data (`benchmark_ground_truth_pipeline.json`) provided in the "annotation" folder. |
| [**FinTagging_BIO**](https://huggingface.co/datasets/TheFinAI/FinTagging_BIO) | BIO-format dataset tailored for token-level tagging with BERT-series models. |

---

## 🧑‍💻 Evaluated LLMs and PLMs
We benchmarked **FinTagging** alongside 10 cutting-edge LLMs and 3 advanced PTMs:

- 🔥 [GPT-4o](https://platform.openai.com/docs/models#gpt-4o)
- 🚀 [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- 🧠 [Qwen2.5 Series](https://huggingface.co/Qwen)
- 📌 [Llama-3 Series](https://huggingface.co/meta-llama)
- 📐 [DeepSeek-R1 Series](https://huggingface.co/deepseek-ai)
- 💡 [Gemma-2 Model](https://huggingface.co/google/gemma-2-27b-it)
- 💰 [Fino1-8B](https://huggingface.co/TheFinAI/Fino1-8B)
- 🏛️ [BERT-large](https://huggingface.co/google-bert/bert-large-uncased)
- 📉 [FinBERT](https://huggingface.co/ProsusAI/finbert)
- 🧾 [SECBERT](https://huggingface.co/nlpaueb/sec-bert-base)

---

## 🎨 Reasoning Path Construction
Inspired by [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1), we provide reasoning paths:
- [**FinCoT**](https://huggingface.co/datasets/TheFinAI/FinCoT)

---

## 🚧 Model Training
We trained **Fino1** using a two-stage pipeline:

1. **🔧 Stage 1 (SFT):** See [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1).
2. **⚙️ Stage 2 (RL - GRPO):** See [open-r1](https://github.com/huggingface/open-r1.git).

---

## 📌 Evaluation Methodology
- **Local Model Inference:** Conducted via [FinBen](https://github.com/The-FinAI/FinBen) (VLLM framework).
- **API-based Model Inference:** Conducted via `query_llm.py` script.
- **Answer Extraction and Evaluation:** Using [DocMath-Eval](https://github.com/yale-nlp/DocMath-Eval).

---

## 📊 Key Performance Metrics

| 📌 Model | 🧮 FinQA | 📑 DocMath-Simplong | 📂 XBRL-Math | 📄 DocMath-Complong | 📈 Avg. |
|----------|----------|---------------------|--------------|----------------------|---------|
| GPT-4o | 72.49 | **60.00** | 72.22 | 39.33 | 61.01 |
| GPT-o1-preview | 49.07 | 56.00 | 74.44 | 36.67 | 54.05 |
| DeepSeek-V3 | 73.20 | 53.00 | 76.67 | **42.33** | **61.30** |
| Fino1-14B | **74.18** | 55.00 | 87.78 | 27.33 | 61.07 |
| Fino1-8B | 73.03 | 56.00 | 84.44 | 26.33 | 59.95 |

*(Selected top-performing models; detailed results in full table.)*

---

## 🗓️ Latest Updates

- **2025-03-30:** 🚀 **Fino1-14B** trained & evaluated.
- **2025-02-12:** 🚀 **Fino1-8B** trained & evaluated.

---

## 📖 Citation

If you find our benchmark useful, please cite:

```bibtex
@misc{qian2025fino1transferabilityreasoningenhanced,
      title={Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance}, 
      author={Lingfei Qian and Weipeng Zhou and Yan Wang and Xueqing Peng and Jimin Huang and Qianqian Xie},
      year={2025},
      eprint={2502.08127},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.08127}, 
}
