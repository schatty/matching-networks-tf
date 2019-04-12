__Omniglot__

| Accuracy                    | 95.9%            | 99.0%            | 88.8%            | 96.1%            |
|-----------------------------|------------------|------------------|------------------|------------------|
| Author                      | Igor Kuznetsov   | Igor Kuznetsov   | Igor Kuznetsov   | Igor Kuznetsov   |
| data.split                  | vinyals          | vinyals          | vinyals          | vinyals          |
| data.train_way              | 5                | 5                | 5                | 5                |
| data.batch                  | 32               | 32               | 32               | 32               |
| data.train_n_support        | 5                | 5                | 5                | 5                |
| data.train_n_query          | 1                | 1                | 1                | 1                |
| data.test_way (val)         | 5                | 5                | 5                | 5                |
| data.test_n_support (val)   | 5                | 5                | 5                | 5                |
| data.test_n_query (val)     | 1                | 1                | 1                | 1                |
| data.train_episodes         | 100              | 100              | 100              | 100              |
| model.x_dim                 | 28,28,1          | 28,28,1          | 28,28,1          | 28,28,1          |
| model.lstm_size             | 32               | 32               | 32               | 32               |
| train.epochs                | 30               | 30               | 30               | 30               |
| train.optim_method          | Adam             | Adam             | Adam             | Adam             |
| train.lr                    | 0.001            | 0.001            | 0.001            | 0.001            |
| train.patience              | 30               | 30               | 30               | 30               |
| data.test_way (test)        | 5                | 5                | 20               | 20               |
| data.test_n_support (test)  | 1                | 5                | 1                | 5                |
| data.test_n_query (test)    | 1                | 5                | 1                | 5                |
| data.test_n_episodes (test) | 100              | 100              | 100              | 100              |
| Encoder CNN architecture    | original (paper) | original (paper) | original (paper) | original (paper) |
| seed                        | 2019             | 2019             | 2019             | 2019             |
