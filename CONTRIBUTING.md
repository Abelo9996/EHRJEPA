# Contributing to JEPA-EHR

Thank you for your interest in contributing to JEPA-EHR! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/EHRJEPA.git
   cd EHRJEPA
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_ehr.txt
   ```

4. **Run tests to verify setup**
   ```bash
   python test_components.py
   ```

## 📋 How to Contribute

### Reporting Bugs

Before submitting a bug report:
- Check existing issues to avoid duplicates
- Use the latest version of the code
- Provide a minimal reproducible example

**Bug Report Template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 2.0.1]

**Additional context**
Add any other context about the problem here.
```

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:
- Clear use case for the enhancement
- Example of how it would work
- Any relevant research papers or references

### Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   python test_components.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Include test results if applicable

## 💻 Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and under 50 lines when possible

**Example:**
```python
def extract_representations(
    data_loader: DataLoader,
    pool: str = 'mean'
) -> np.ndarray:
    """
    Extract representations from the encoder.
    
    Args:
        data_loader: DataLoader with EHR sequences
        pool: Pooling method ('mean', 'max', 'last')
    
    Returns:
        Extracted representations as numpy array
    """
    # Implementation here
    pass
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests liberally

**Good examples:**
```
Add support for hierarchical transformers
Fix bug in temporal masking for edge cases
Update README with new installation instructions
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python test_components.py

# Test specific components
python -m pytest tests/test_models.py -v
```

### Writing Tests

- Add tests for all new features
- Ensure tests are deterministic
- Use descriptive test names

**Example:**
```python
def test_temporal_transformer_forward_pass():
    """Test that temporal transformer produces expected output shape"""
    model = TemporalTransformer(
        num_features=25,
        sequence_length=20,
        embed_dim=384
    )
    x = torch.randn(8, 20, 25)  # batch_size, seq_len, features
    output = model(x)
    assert output.shape == (8, 20, 384)
```

## 📚 Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include examples for complex functions

### README Updates

When adding new features:
- Update the relevant section in README.md
- Add usage examples
- Update the project structure if new files are added

## 🏥 Working with MIMIC-IV Data

**Important**: Never commit actual MIMIC-IV data to the repository!

- The `.gitignore` excludes `mimic-iv-2.1/` and `data/` directories
- Use synthetic data for tests and examples
- When sharing examples, use `generate_sample_data_clean.py`

## 🎯 Priority Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Improved preprocessing pipelines for MIMIC-IV
- [ ] Additional downstream tasks (sepsis prediction, AKI prediction, etc.)
- [ ] Multi-GPU training support
- [ ] Comprehensive unit tests

### Medium Priority
- [ ] Attention visualization tools
- [ ] Hyperparameter tuning utilities
- [ ] Additional masking strategies
- [ ] Integration with MLflow or Weights & Biases

### Low Priority
- [ ] Web demo or visualization dashboard
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Performance benchmarking suite

## 📝 License

By contributing to JEPA-EHR, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 📞 Questions?

If you have questions about contributing:
- Open a GitHub issue with the "question" label
- Check existing documentation and issues first
- Be specific and provide context

## 🙏 Recognition

Contributors will be recognized in:
- The project README
- Release notes for significant contributions
- Special thanks in the paper/documentation

Thank you for contributing to JEPA-EHR! 🎉
