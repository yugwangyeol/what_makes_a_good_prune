# What Makes a Good Prune?

<div align="center">
  <img src="https://github.com/user-attachments/assets/9deb18d7-66ff-4c77-8cf1-5e45bd600568" alt="What Makes a Good Prune">
</div>

**What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity**는 2024년 ICLR에서 발표된 논문으로, 신경망의 **Unstructured Pruning**을 평가하고 수행하는 새로운 접근법을 제안하였습니다.  
이 논문의 주요 초점은 **Cosine Similarity**를 활용하여, 원본 모델과 Pruned 모델의 표현 능력을 최대한 유지하면서 복잡성을 줄이는 것입니다.  

기존의 **L1 노름 기반 Pruning**이나 **Random Pruning**과는 다르게, 이 접근법은 **파라미터 벡터의 방향성**을 보존하는 데 중점을 둡니다.  
이 방식은 Pruning 후에도 모델이 원본 네트워크의 기능적 특성을 더 잘 유지하도록 설계되었습니다.  
해당 논문에서는 모델의 정확도, Parameter Size, Inference Time 등을 측정하고 기존 방법들과 비교하는 작업을 진행하였습니다.

---

## Citation

```bibtex
@inproceedings{
  mason-williams2024what,
  title={What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity},
  author={Gabryel Mason-Williams and Fredrik Dahlqvist},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=jsvvPVVzwf}
}
