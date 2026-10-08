<div align="center">

<h1>FinTagging</h1>

<p><strong>Benchmarking LLMs for Extracting and Structuring Financial Information</strong></p>

<p>
  <a href="https://arxiv.org/abs/2505.20650"><img src="https://img.shields.io/badge/arXiv-2505.20650-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/TheFinAI/fintagging-xbrl-tagging-68270132372c6608ac069bef"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-yellow?logo=huggingface" alt="Hugging Face Collection"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

<p>
  <a href="https://arxiv.org/abs/2505.20650">Paper</a> ·
  <a href="https://huggingface.co/collections/TheFinAI/fintagging-xbrl-tagging-68270132372c6608ac069bef">Data</a> ·
  <a href="https://github.com/The-FinAI/FinBen">Evaluation Framework</a>
</p>

</div>

---

## Overview

**FinTagging** is an LLM-ready benchmark for structure-aware, full-scope XBRL tagging: mapping the numerical facts in financial reports to concepts in the US-GAAP taxonomy. It decomposes tagging into two subtasks: **FinNI** (Financial Numeric Identification), which extracts numeric entities and their types from text and tables, and **FinCL** (Financial Concept Linking), which links each extracted entity to the full US-GAAP taxonomy. This repository contains the annotations, taxonomy files, retrieval and BERT baseline scripts, and the notebooks used to build and evaluate the benchmark.

## Resources on Hugging Face

All FinTagging data is collected in the [FinTagging Hugging Face collection](https://huggingface.co/collections/TheFinAI/fintagging-xbrl-tagging-68270132372c6608ac069bef).

| Dataset | Description |
|---------|-------------|
| [**TheFinAI/en-finni-eval**](https://huggingface.co/datasets/TheFinAI/en-finni-eval) (formerly `FinNI-eval`) | Evaluation set for the FinNI subtask of the FinTagging benchmark. |
| [**TheFinAI/en-fincl-eval**](https://huggingface.co/datasets/TheFinAI/en-fincl-eval) (formerly `FinCL-eval`) | Evaluation set for the FinCL subtask of the FinTagging benchmark. |
| [**TheFinAI/en-fintagging-original**](https://huggingface.co/datasets/TheFinAI/en-fintagging-original) (formerly `FinTagging_Original`) | Original benchmark dataset without preprocessing, suitable for custom research. The annotated data (`benchmark_ground_truth_pipeline.json`) is provided in the [`annotation/`](annotation) folder. |
| [**TheFinAI/en-fintagging-bio**](https://huggingface.co/datasets/TheFinAI/en-fintagging-bio) (formerly `FinTagging_BIO`) | BIO-format dataset tailored for token-level tagging with BERT-series models. The same data is provided in the [`BERT/data`](BERT/data) folder as `test_data_benchmark.bio`. |

| Model | Description |
|-------|-------------|
| [**TheFinAI/Fino1-8B**](https://huggingface.co/TheFinAI/Fino1-8B) | Our in-house financial reasoning LLM, evaluated on FinTagging. |

### Data in this repository

| Data | Description |
|------|-------------|
| [**FinTagging_Trainset**](annotation/TrainingSet_Annotation.json) | Training set for the BERT-series models, provided in two formats: JSON (`annotation/TrainingSet_Annotation.json`) and BIO (`BERT/data/train_data_all.bio`). |
| [**FinTagging_Subset**](subdata) | Subsets for the FinNI and FinCL tasks (`subdata/`). |
| [**Taxonomy**](taxonomy) | The original US-GAAP taxonomy file (`us-gaap-2024.xsd`) and the processed taxonomy BM25 index document (`us_gaap_2024_BM25.jsonl`). |

## Evaluated LLMs and PLMs

We benchmarked **FinTagging** with 10 cutting-edge LLMs and 3 advanced PLMs:

- **[GPT-4o](https://platform.openai.com/docs/models#gpt-4o)**: OpenAI's multimodal flagship model with structured output support.
- **[DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)**: a MoE reasoning model with efficient inference via MLA.
- **[Qwen2.5 Series](https://huggingface.co/Qwen)**: multilingual models optimized for reasoning, coding, and math. We assessed the 14B, 1.5B, and 0.5B Instruct models.
- **[Llama-3 Series](https://huggingface.co/meta-llama)**: Meta's open-source instruction-tuned models for long context. We assessed Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct.
- **[DeepSeek-R1 Series](https://huggingface.co/deepseek-ai)**: RL-tuned first-generation reasoning models with zero-shot strength. We assessed DeepSeek-R1-Distill-Qwen-32B.
- **[Gemma-2](https://huggingface.co/google/gemma-2-27b-it)**: Google's instruction-tuned model with open weights. We assessed gemma-2-27b-it.
- **[Fino1-8B](https://huggingface.co/TheFinAI/Fino1-8B)**: our in-house financial LLM with strong reasoning capability.
- **[BERT-large](https://huggingface.co/google-bert/bert-large-uncased)**: the classic transformer encoder for language understanding.
- **[FinBERT](https://huggingface.co/ProsusAI/finbert)**: a financial domain-tuned BERT for sentiment analysis.
- **[SECBERT](https://huggingface.co/nlpaueb/sec-bert-base)**: a BERT model trained on SEC filings for financial disclosure tasks.

## Evaluation

- **Local model inference** is run through [FinBen](https://github.com/The-FinAI/FinBen) (vLLM framework).
- Task-specific evaluation scripts are provided in our fork of the FinBen framework: <https://github.com/Yan2266336/FinBen>.
- **FinNI:** run the provided script directly to evaluate a variety of LLMs, including both local and API-based models.
- **FinCL:** first run the retrieval script in this repository ([`retrieval/`](retrieval)) to obtain US-GAAP candidate concepts. Then use our prompts to construct instruction-style inputs, and apply the reranking method implemented in the forked FinBen to identify the most appropriate US-GAAP concept.
- **Taxonomy:** the original US-GAAP taxonomy file (`us-gaap-2024.xsd`) and the processed taxonomy BM25 index document (`us_gaap_2024_BM25.jsonl`) are in the [`taxonomy/`](taxonomy) folder.

> [!NOTE]
> Running the retrieval script requires a local installation of Elasticsearch. Our embedding index document is available on [Google Drive](https://drive.google.com/file/d/1cyMONjP9WdHtD8-WGezmgh_LNhbY3qtR/view?usp=drive_link). You can also build your own index document from the original US-GAAP taxonomy file instead of using ours.

## Results

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

<div style="font-size: 10px; overflow-x: auto; width: 100%;">
  <table>
    <caption><strong>Table: The FinNI Task Performance</strong><br>
    <em>🥇 = best, 🥈 = second-best, 🥉 = third-best</em>
    </caption>
    <thead>
      <tr>
        <th>Category</th>
        <th>Models</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Closed-source LLM</td>
        <td>GPT-4o</td>
        <td>0.6105 🥈</td>
        <td>0.5941 🥈</td>
        <td>0.6022 🥈</td>
      </tr>
      <tr>
        <td rowspan="8">Open-source LLMs</td>
        <td>DeepSeek-V3</td>
        <td>0.6329 🥇</td>
        <td>0.8452 🥇</td>
        <td>0.7238 🥇</td>
      </tr>
      <tr>
        <td>DeepSeek-R1-Distill-Qwen-32B</td>
        <td>0.5490 🥉</td>
        <td>0.2238 🥉</td>
        <td>0.3180 🥉</td>
      </tr>
      <tr>
        <td>Qwen2.5-14B-Instruct</td>
        <td>0.3632</td>
        <td>0.0018</td>
        <td>0.0035</td>
      </tr>
      <tr>
        <td>gemma-2-27b-it</td>
        <td>0.5319</td>
        <td>0.5490 🥉</td>
        <td>0.5403 🥉</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B-Instruct</td>
        <td>0.3346</td>
        <td>0.1746</td>
        <td>0.2295</td>
      </tr>
      <tr>
        <td>Llama-3.2-3B-Instruct</td>
        <td>0.1887</td>
        <td>0.1794</td>
        <td>0.1839</td>
      </tr>
      <tr>
        <td>Qwen2.5-1.5B-Instruct</td>
        <td>0.1323</td>
        <td>0.0636</td>
        <td>0.0859</td>
      </tr>
      <tr>
        <td>Qwen2.5-0.5B-Instruct</td>
        <td>0.0116</td>
        <td>0.0027</td>
        <td>0.0043</td>
      </tr>
      <tr>
        <td>Financial LLM</td>
        <td>Fino1-8B</td>
        <td>0.3416</td>
        <td>0.1481</td>
        <td>0.2066</td>
      </tr>
    </tbody>
  </table>
</div>


---

<div style="font-size: 10px; overflow-x: auto; width: 100%;">
  <table>
    <caption><strong>Table: The FinCL Task Performance</strong><br>
    <em>🥇 = best, 🥈 = second-best, 🥉 = third-best</em>
    </caption>
    <thead>
      <tr>
        <th>Category</th>
        <th>Models</th>
        <th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Closed-source LLM</td>
        <td>GPT-4o</td>
        <td>0.1664 🥈</td>
      </tr>
      <tr>
        <td rowspan="8">Open-source LLMs</td>
        <td>DeepSeek-V3</td>
        <td>0.1715 🥇</td>
      </tr>
      <tr>
        <td>DeepSeek-R1-Distill-Qwen-32B</td>
        <td>0.1013</td>
      </tr>
      <tr>
        <td>Qwen2.5-14B-Instruct</td>
        <td>0.1072 🥉</td>
      </tr>
      <tr>
        <td>gemma-2-27b-it</td>
        <td>0.1009</td>
      </tr>
      <tr>
        <td>Llama-3.1-8B-Instruct</td>
        <td>0.0807</td>
      </tr>
      <tr>
        <td>Llama-3.2-3B-Instruct</td>
        <td>0.0375</td>
      </tr>
      <tr>
        <td>Qwen2.5-1.5B-Instruct</td>
        <td>0.0419</td>
      </tr>
      <tr>
        <td>Qwen2.5-0.5B-Instruct</td>
        <td>0.0246</td>
      </tr>
      <tr>
        <td>Financial LLM</td>
        <td>Fino1-8B</td>
        <td>0.0704</td>
      </tr>
    </tbody>
  </table>
</div>

## Citation

If you find our benchmark useful, please cite:

```bibtex
@misc{wang2025fintaggingbenchmarkingllmsextracting,
      title={FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information}, 
      author={Yan Wang and Yang Ren and Lingfei Qian and Xueqing Peng and Keyi Wang and Yi Han and Dongji Feng and Fengran Mo and Shengyuan Lin and Qinchuan Zhang and Kaiwen He and Chenri Luo and Jianxing Chen and Junwei Wu and Jimin Huang and Guojun Xiong and Xiao-Yang Liu and Qianqian Xie and Jian-Yun Nie},
      year={2025},
      eprint={2505.20650},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20650}, 
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). Datasets and models on Hugging Face keep their own licenses, stated on each card.

---

<p align="center">Built by <a href="https://thefin.ai">The Fin AI</a> · <a href="https://huggingface.co/TheFinAI">Hugging Face</a> · <a href="https://github.com/The-FinAI">GitHub</a></p>
