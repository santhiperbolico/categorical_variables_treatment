# Categorical Variables Treatment

This Python package provides tools for handling categorical variables in machine learning tasks, with a focus on embedding techniques.

## Features

- One-hot encoding implementation
- Neural network-based embedding using Keras
- LLM (Language Model) embedding with PCA dimensionality reduction
- Comparative analysis of different embedding methods

## Installation

```bash
pip install categorical-variables-treatment
```

## Usage

Here's a quick example of how to use the package:

```python
from categorical_variables_treatment import OneHotEncoder, NeuralEmbedding, LLMEmbedding

# Load your data
df = pd.read_csv('your_data.csv')

# One-hot encoding
one_hot = OneHotEncoder()
df_onehot = one_hot.fit_transform(df['ORGANIZATION_TYPE'])

# Neural network-based embedding
neural_emb = NeuralEmbedding(embedding_dim=5)
df_neural = neural_emb.fit_transform(df['ORGANIZATION_TYPE'])

# LLM embedding with PCA
llm_emb = LLMEmbedding(n_components=5)
df_llm = llm_emb.fit_transform(df['ORGANIZATION_TYPE'])
```

## Documentation

For more detailed information about the package and its functions, please refer to the [documentation](https://github.com/santhiperbolico/categorical_variables_treatment/wiki).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
