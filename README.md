<!-- <p align="center" width="100%">
<img src="./docs/static/images/logo_resize.png"  width="80%">
</p> -->

<div align="center">
    <h1 align="center"> SIM-CoT: Supervised Implicit Chain-of-Thought
    </h1>
</div>


<a href="">
<img src='https://img.shields.io/badge/Project-Website-orange' alt='Project Page'></a>
<a href="">
<img src='https://img.shields.io/badge/arXiv-SIM_CoT-blue' alt='Paper PDF'></a>


<a href="https://huggingface.co/internlm/SIM_COT-LLaMA3-CODI-8B">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20LLaMA%208b%20CODI-yellow">
</a>
<a href="">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20LLaMA%203b%20CODI-yellow">
</a>
<a href="">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20LLaMA%201b%20CODI-yellow">
</a>
<a href="">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20GPT2%20CODI-yellow">
</a>


<a href="">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20LLaMA%201b%20Coconut-yellow">
</a>
<a href="">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20SIM_CoT%20GPT2%20Coconut-yellow">
</a>

----

- **Authors**: [Xilin Wei](https://github.com/Wiselnn570), [Xiaoran Liu](https://scholar.google.de/citations?user=Qe6F4J4AAAAJ&hl=en), [Yuhang Zang](https://yuhangzang.github.io), [Xiaoyi Dong](https://lightdxy.github.io), [Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ&hl=en), [Jiaqi Wang](https://myownskyw7.github.io/), [Xipeng Qiu](https://xpqiu.github.io/en.html), [Dahua Lin](http://dahua.site/)
- **Institutes**: Fudan University; Shanghai AI Laboratory; The Chinese University of Hong Kong; Shanghai Innovation Institute; 

## 💡 Highlights

- 🔥 **Latent Instability in Implicit CoT:** We systematically analyze the limitations of implicit Chain-of-Thought methods and reveal a **latent instability issue**—as the number of implicit tokens increases, models tend to collapse into homogeneous latent states that lose operator semantics.  

- 🔥 **Step-Level Supervision with SIM-CoT:** We propose **S**upervised **IM**plicit-CoT (**SIM-CoT**), a plug-and-play module that introduces **step-level supervision** via an auxiliary decoder. This stabilizes optimization, prevents collapse, and ensures that latent tokens capture meaningful reasoning steps.

- 🔥 **Strong and Consistent Performance:** SIM-CoT consistently outperforms both explicit and implicit baselines. On GPT-2, it exceeds supervised CoT by **+2.1%**, Coconut by **+8.2%**, and CODI by **+4.3%**. Across larger LLaMA models (1B/3B/8B), it delivers **+1.5% to +9.0%** gains, and remains stable even with **8–16 implicit tokens**, where prior methods collapse.  

- 🔥 **Efficiency and Interpretability:** SIM-CoT adds **no extra inference cost** since the auxiliary decoder is discarded after training. It also provides **interpretability**, allowing each latent token to be decoded into a human-readable reasoning step.  

## 📜 News

**[2025/9/24]** [Code]() and [Paper]() are released!

## 👨‍💻 Todo

- [x] Code Release
- [ ] Checkpoint Release
- [ ] Usage Instructions Release


<!-- ## 🛠️ Usage
- Required Package Versions
  ```
  transformers 4.45.2
  vllm 0.6.3.post2.dev171+g890ca360
  ```

- The implementation of videorope (both transformers and vllm) is emphasized with **#!**, and you can easily find it by pressing ctrl + F.
- For transformer inference:
  ```
  with torch.inference_mode():
      generated_ids = model.generate(
        ..., 
        which_rope=which_rope,
        scale_factor=scale_factor
      )
      generated_ids_trimmed = [
          out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
      ]
      output_text = processor.batch_decode(
          generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
      )
      generated_text = output_text[0]
  ```
- For vLLM inference:
  ```
  mm_data['which_rope'] = which_rope
  mm_data['scale_factor'] = scale_factor
  llm_inputs = {
      "prompt": prompt,
      "multi_modal_data": mm_data,
  }
  with torch.no_grad():
      outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
  generated_text = outputs[0].outputs[0].text
  ``` -->
<!-- ## Quick Usage

You can directly use our ShareGPT4Video model for conversation with your own video by the following command:

```
python run.py --model-path Lin-Chen/sharegpt4video-8b --video examples/yoga.mp4 --query Describe this video in detail.
```

Or you can build your local demo to enjoy our ShareGPT4Video-8B with the following command:

```
python app.py
```

You can build your local demo for enjoying our ShareCaptioner-Video with the following command:

```
cd captioner

python app.py
```

## Install

```bash
git clone https://github.com/ShareGPT4Omni/ShareGPT4Video
conda create -n share4video python=3.10 -y
conda activate share4video

cd ShareGPT4Video
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
 -->

## ✒️ Citation

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex
@article{wei2025simcot,
  title={{SIM-COT}: Supervised Implicit Chain-of-Thought},
  author={Wei, Xilin and Liu, Xiaoran and Zang, Yuhang and Dong, Xiaoyi and Cao, Yuhang and Wang, Jiaqi and Qiu, Xipeng and Lin, Dahua},
  journal={arXiv preprint arXiv:2509.20317},
  year={2025}
}
```

## ❤️ Acknowledgments

- [Coconut](https://github.com/facebookresearch/coconut): The codebase we built upon. Thanks for their wonderful work.
- [CODI](https://github.com/zhenyi4/codi): Our work is based on this codebase; we are grateful for their valuable contribution.
- [LLaMA series](https://huggingface.co/meta-llama/collections): The amazing open-sourced large language model!
- [GPT2](https://huggingface.co/openai-community/gpt2): An impressive open-source large language model!
