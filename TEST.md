# Test mode

## Test DB

```bash
docker-compose -f tests/docker/mysql.yml up -d
```

```python
# MODE in ['test', 'dev']
conda activate avt

MODE=test pytest
MODE=test pytest tests/test_connection.py --log-cli-level=INFO


# generate database
MODE=test python tests/db_gen.py

```
