# what_makes_a_good_prune

![image](https://github.com/user-attachments/assets/9deb18d7-66ff-4c77-8cf1-5e45bd600568)

What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity"는 2024년 ICLR에 발표된 논문으로, 신경망의 unstructured pruning을 평가하고 수행하는 새로운 접근법을 제안했다. 이 논문의 주요 초점은 cosine similarity를 활용하여 원본 모델과 Pruning 모델의 표현 능력을 최대한 유지하면서 복잡성을 줄이는 것이다. 기존의 L1 노름 기반 Pruning 나 Random Pruning과 달리, 이 방법은 파라미터 벡터의 방향성을 보존하는 데 중점을 둔다. 이는 Pruning 후에도 모델이 원본 네트워크의 기능적 특성을 더 잘 유지하도록 한다. 해당 모델에 대해 정확도, Parameter size, Inference time을 측정하고 비교하는 작업 진행하였다.

## Citation

```
@inproceedings{
  mason-williams2024what,
  title={What Makes a Good Prune? Maximal Unstructured Pruning for Maximal Cosine Similarity},
  author={Gabryel Mason-Williams and Fredrik Dahlqvist},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=jsvvPVVzwf }
}
```