## Description

This is the supplementary code repo for the paper ["Learning Syntax from Naturally-Occurring Bracketings" (NAACL 2021)](https://www.aclweb.org/anthology/2021.naacl-main.234.pdf).

## Dependencies

The code has been tested with the following dependencies and versions:

```
python==3.6.7
torch==1.7.1
transformers==4.4.0
numpy==1.19.4
fire==0.1.3
```

## Data

Our pre-processed data are included in the `data` directory. You can use the command `tar xzvf data.tar.gz` to decompress.

## Training the model

Simply run `./train.sh`. You can change the training data source and the loss function through the `DATASOURCE` and `CHART_MODE` variables.

## Evaluation

Run `python evaluate.py`. Change model path and test file path as needed.

## License

Our code is based on [this project](https://github.com/tzshi/flat-mwe-parsing) and licensed under MIT license.
The file `attention.py` is based on an implemention in [AllenNLP](https://github.com/allenai/allennlp)(Apache-2.0).

## Reference

You can cite our paper if our project is useful to your research:

Tianze Shi, Ozan İrsoy, Igor Malioutov, and Lillian Lee. 2021. Learning Syntax from Naturally-Occurring Bracketings.
In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics_.
