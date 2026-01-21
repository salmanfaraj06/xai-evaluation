## Pull Request Checklist

Please ensure your PR meets the following requirements:

### Code Quality
- [ ] Code follows project style guidelines
- [ ] No linting errors (`flake8 hexeval/ tests/`)
- [ ] Code is formatted with Black (`black hexeval/ tests/`)
- [ ] No unused imports or variables

### Testing
- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Test coverage maintained or improved
- [ ] Manual testing completed

### Documentation
- [ ] Code is well-commented
- [ ] Docstrings added for new functions/classes
- [ ] README updated if needed
- [ ] CHANGELOG updated (if applicable)

### Functionality
- [ ] Changes work as expected
- [ ] No breaking changes (or documented if necessary)
- [ ] Streamlit app runs without errors
- [ ] Demo use cases still work

### Security
- [ ] No secrets or API keys committed
- [ ] Dependencies are secure and up-to-date
- [ ] No sensitive data in code or tests

## Description

<!-- Provide a brief description of your changes -->

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

## Screenshots (if applicable)

<!-- Add screenshots to help explain your changes -->

## Additional Notes

<!-- Any additional information that reviewers should know -->
