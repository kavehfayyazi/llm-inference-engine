# Running the Triton kernel on Colab (CUDA)

The Triton paged-decode kernel needs CUDA. Mac/MPS runs the reference path only.
Use a Colab GPU runtime (Runtime -> Change runtime type -> GPU; free tier is a T4).

Run each cell:

```python
# 1. Clone the repo
!git clone https://github.com/kavehfayyazi/llm-inference-engine.git
%cd llm-inference-engine
```

```python
# 2. Deps. torch on Colab ships with Triton already; transformers for the model.
!pip -q install transformers accelerate
```

```python
# 3. Gated model: log in with an HF token that has Llama-3.2 access
from huggingface_hub import login
login()  # paste token
```

```python
# 4. Sanity: confirm CUDA + the reference path passes on GPU first
import torch; print("cuda:", torch.cuda.is_available())
!PYTHONPATH=. python tests/test_reference.py
```

```python
# 5. The real check: triton kernel must match reference + HF
!PYTHONPATH=. python scripts/verify_triton.py
```

Expected final output:

```
TRITON OK: kernel matches reference + HF on 4 prompts
```

If step 5 fails, the assert prints which prompt diverged. The reference path (step 4)
is the oracle: if step 4 passes but step 5 fails, the bug is in the kernel, not the
engine. Paste the error back and we iterate.
```
