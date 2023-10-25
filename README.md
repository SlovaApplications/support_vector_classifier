# Support Vector Classifier for Text

## Getting Started
**NOTE:** This repo was created and tested using Python 3.12. It may work with earlier versions of Python 3, but that is not guaranteed.

1. Install the dependencies for this library with the following command

    ```pip install -r requirements.txt```

2. Prepare your training data in a `.jsonl` file. The file should have one JSON object per row. Each JSON object should have two keys, `text` and `include`. The value of the `text` key should be a string. The value of the `include` key should be either `1` or `0`, indicating whether the text should be classified as 'included' or 'excluded'. See the `example_data/training_data.jsonl` file for an example of this.

3. Train the classifier using the following command the training data you prepared in Step 2.

    ```python train_support_vector_classifier.py {path to training data} {path where model will be saved}```

    > **Example**
    > 
    > ```python train_support_vector_classifier.py example_data/training_data.jsonl svm_model.pickle```

4. Prepare new data for classification. Similar to the training data, the data you want to classify should be formatted as a `.jsonl` file. The file should have one JSON object per row. Each JSON object should have one key, `text`. The value of that key should be a string. See the `example_data/data_to_classify.jsonl` for an example of this.

5. Classify the data prepared in Step 4 by running the following command:

    ```python classify_records.py {path to model created in Step 3} {path to data prepared in Step 4} {path where the classified data will be saved}```

    > **Example**
    > 
    > ```python classify_records.py svm_model.pickle example_data/data_to_classify.jsonl classified_data.jsonl```
 
6. Examine the results of your output file from Step 5.

# Citation

The sample data was sourced from Stanford's Large Movie Review Dataset.

```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
