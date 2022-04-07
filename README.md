# Code for [Few-shot Unsupervised Domain Adaptation with Image-to-Class Sparse Similarity Encoding](https://arxiv.org/abs/2108.02953)
## Usage: Few-shot Unsupervised Domain Adaptation
* Execute the below command to train a IMSE model.
```shell
python run_script.py --gpu=GIDS --metric=IMSE
```
* Check results/saved model weights in "./adjust_parameters"

* If you want to add some new methods, please set it as below

```python
import model.hyper_model

model.hyper_model.Trainer.method.classifier = YOURMETHOD
```
## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{
  author    = {Shengqi Huang and
               Wanqi Yang and
               Lei Wang and
               Luping Zhou and
               Ming Yang},
  title     = {Few-shot Unsupervised Domain Adaptation with Image-to-Class Sparse
               Similarity Encoding},
  booktitle = {{MM} '21: {ACM} Multimedia Conference, Virtual Event, China, October
               20 - 24, 2021},
  pages     = {677--685},
  publisher = {{ACM}},
  year      = {2021},
  url       = {https://doi.org/10.1145/3474085.3475232},
  doi       = {10.1145/3474085.3475232},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```