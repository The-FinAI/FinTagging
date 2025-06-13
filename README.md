<!-- Title -->
<h1 align="center">✨ FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information ✨</h1>

<p align="center">
  📁 <a href="https://huggingface.co/datasets/TheFinAI/FinTagging_Original">Benchmark Data</a> | 📖 <a href="https://arxiv.org/abs/2505.20650">Arxiv</a> | 🛠️ <a href="https://github.com/The-FinAI/FinBen">Evaluation Framework</a>
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
| [**FinTagging_Trainset**](https://huggingface.co/datasets/TheFinAI/FinTagging_BIO) | BIO-format dataset tailored for token-level tagging with BERT-series models. |

---

## 🧑‍💻 Evaluated LLMs and PLMs
We benchmarked **FinTagging** alongside 10 cutting-edge LLMs and 3 advanced PLMs:

- 🌐 **[GPT-4o](https://platform.openai.com/docs/models#gpt-4o)** — OpenAI’s multimodal flagship model with structured output support.
- 🚀 **[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)** — A MoE reasoning model with efficient inference via MLA.
- 🧠 **[Qwen2.5 Series](https://huggingface.co/Qwen)** — Multilingual models optimized for reasoning, coding, and math. Here, we assessed 14B, 1.5B, and 0.8B Instruct models.
- 🦙 **[Llama-3 Series](https://huggingface.co/meta-llama)** — Meta’s open-source instruction-tuned models for long context. Here, we assessed the Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct models.
- 🧭 **[DeepSeek-R1 Series](https://huggingface.co/deepseek-ai)** — RL-tuned first-gen reasoning models with zero-shot strength. Here, we only assessed the DeepSeek-R1-Distill-Qwen-32B model.
- 🧪 **[Gemma-2 Model](https://huggingface.co/google/gemma-2-27b-it)** — Google’s latest instruction-tuned model with open weights. Here, we only assess the gemma-2-27b-it model.
- 💎 **[Fino1-8B](https://huggingface.co/TheFinAI/Fino1-8B)** — Our in-house financial LLM with strong reasoning capability.
- 🏛️ **[BERT-large](https://huggingface.co/google-bert/bert-large-uncased)** — The classic transformer encoder for language understanding.
- 📉 **[FinBERT](https://huggingface.co/ProsusAI/finbert)** — A financial domain-tuned BERT for sentiment analysis.
- 🧾 **[SECBERT](https://huggingface.co/nlpaueb/sec-bert-base)** — BERT model fine-tuned on SEC filings for financial disclosure tasks.


---

## 📌 Evaluation Methodology
- **Local Model Inference:** Conducted via [FinBen](https://github.com/The-FinAI/FinBen) (VLLM framework).
- We provide task-specific evaluation scripts through our forked version of the FinBen framework, available at: https://github.com/Yan2266336/FinBen.
- For the FinNI task, you can directly execute the provided script to evaluate a variety of LLMs, including both local and API-based models.
- For the FinCL task, first run the retrieval script from the repository to obtain US-GAAP candidate concepts. Then, use our provided prompts to construct instruction-style inputs, and apply the reranking method implemented in the forked FinBen to identify the most appropriate US-GAAP concept.
- **Note**: Running the retrieval script requires a local installation of Elasticsearch, we provided our index document at Google Drive: https://drive.google.com/file/d/1cyMONjP9WdHtD8-WGezmgh_LNhbY3qtR/view?usp=drive_link. However, you can construct your own index document instead of using ours.

---

## 📊 Key Performance Metrics

<div style="font-size: 10px; overflow-x: auto; width: 100%;">
  <table>
    <caption><strong>Table: Overall Performance</strong><br>
    <em>🥇 = best, 🥈 = second-best, 🥉 = third-best</em>
    </caption>
    <thead>
      <tr>
        <th>Category</th>
        <th>Models</th>
        <th>Macro P</th>
        <th>Macro R</th>
        <th>Macro F1</th>
        <th>Micro P</th>
        <th>Micro R</th>
        <th>Micro F1</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Closed-source LLM</td>
        <td>GPT-4o</td>
        <td>0.0764 🥈</td>
        <td>0.0576 🥈</td>
        <td>0.0508 🥈</td>
        <td>0.0947</td>
        <td>0.0788</td>
        <td>0.0860</td>
      </tr>
      <tr>
        <td rowspan="8">Open-source LLMs</td>
        <td>DeepSeek-V3</td>
        <td>0.0813 🥇</td>
        <td>0.0696 🥇</td>
        <td>0.0582 🥇</td>
        <td>0.1058</td>
        <td>0.1217 🥉</td>
        <td>0.1132 🥉</td>
      </tr>
      <tr>
        <td>DeepSeek-R1-Distill-Qwen-32B</td>
        <td>0.0482 🥉</td>
        <td>0.0288 🥉</td>
        <td>0.0266 🥉</td>
        <td>0.0692</td>
        <td>0.0223</td>
        <td>0.0337</td>
      </tr>
      <tr>
        <td>Qwen2.5-14B-Instruct</td>
        <td>0.0423</td>
        <td>0.0256</td>
        <td>0.0235</td>
        <td>0.0197</td>
        <td>0.0133</td>
        <td>0.0159</td>
      </tr>
      <tr>
        <td>gemma-2-27b-it</td>
        <td>0.0430</td>
        <td>0.0273</td>
        <td>0.0254</td>
        <td>0.0519</td>
        <td>0.0453</td>
        <td>0.0483</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B-Instruct</td>
        <td>0.0287</td>
        <td>0.0152</td>
        <td>0.0137</td>
        <td>0.0462</td>
        <td>0.0154</td>
        <td>0.0231</td>
      </tr>
      <tr>
        <td>Llama-3.2-3B-Instruct</td>
        <td>0.0182</td>
        <td>0.0109</td>
        <td>0.0083</td>
        <td>0.0151</td>
        <td>0.0102</td>
        <td>0.0121</td>
      </tr>
      <tr>
        <td>Qwen2.5-1.5B-Instruct</td>
        <td>0.0180</td>
        <td>0.0079</td>
        <td>0.0069</td>
        <td>0.0248</td>
        <td>0.0060</td>
        <td>0.0096</td>
      </tr>
      <tr>
        <td>Qwen2.5-0.5B-Instruct</td>
        <td>0.0014</td>
        <td>0.0003</td>
        <td>0.0004</td>
        <td>0.0047</td>
        <td>0.0001</td>
        <td>0.0002</td>
      </tr>
      <tr>
        <td>Financial LLM</td>
        <td>Fino1-8B</td>
        <td>0.0299</td>
        <td>0.0146</td>
        <td>0.0140</td>
        <td>0.0355</td>
        <td>0.0133</td>
        <td>0.0193</td>
      </tr>
      <tr>
        <td rowspan="3">Fine-tuned PLMs</td>
        <td>BERT-large</td>
        <td>0.0135</td>
        <td>0.0200</td>
        <td>0.0126</td>
        <td>0.1397 🥈</td>
        <td>0.1145 🥈</td>
        <td>0.1259 🥈</td>
      </tr>
      <tr>
        <td>FinBERT</td>
        <td>0.0088</td>
        <td>0.0143</td>
        <td>0.0087</td>
        <td>0.1293 🥉</td>
        <td>0.0963</td>
        <td>0.1104</td>
      </tr>
      <tr>
        <td>SECBERT</td>
        <td>0.0308</td>
        <td>0.0483</td>
        <td>0.0331</td>
        <td>0.2144 🥇</td>
        <td>0.2146 🥇</td>
        <td>0.2145 🥇</td>
      </tr>
    </tbody>
</table>
</div>



---

## 📖 Citation

If you find our benchmark useful, please cite:

```bibtex
@misc{wang2025fintaggingllmreadybenchmarkextracting,
      title={FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information}, 
      author={Yan Wang and Yang Ren and Lingfei Qian and Xueqing Peng and Keyi Wang and Yi Han and Dongji Feng and Xiao-Yang Liu and Jimin Huang and Qianqian Xie},
      year={2025},
      eprint={2505.20650},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20650}, 
}
