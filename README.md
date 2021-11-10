#Few-shot framework

* Run command below to train a FSL/FSUDA model.
```shell
python run_script.py --gpu=GIDS --metric=IMSE
```
* Check results/saved model weights in "./adjust_parameters"

* If you want to add some new methods, please set it as below

```python
import model.hyper_model

model.hyper_model.Trainer.method.classifier = YOURMETHOD
```