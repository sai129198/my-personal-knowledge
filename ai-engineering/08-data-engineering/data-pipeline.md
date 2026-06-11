#topic/data-engineering #topic/pipeline #topic/etl #year/2026 #status/draft

# 数据 Pipeline 设计与工具

> **一句话定位**：构建从原始数据到模型可用数据的自动化流水线，确保数据流的可靠性、可观测性和可复现性。

---

## 1. 数据 Pipeline 架构

### 1.1 典型 AI 数据 Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  数据采集   │ → │  数据清洗   │ → │  数据转换   │ → │  数据存储   │
│  (Ingest)   │    │  (Clean)    │    │  (Transform)│    │  (Store)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       ↑                                                    │
       └──────────────── 监控 & 调度 ────────────────────────┘
```

### 1.2 Pipeline 类型

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| **批处理 (Batch)** | 定时执行、处理大量历史数据 | 模型训练数据准备 |
| **流处理 (Streaming)** | 实时处理、低延迟 | 在线推理数据预处理 |
| **混合 (Hybrid)** | Lambda/Kappa 架构 | 实时+离线结合 |

---

## 2. 核心组件设计

### 2.1 数据采集层

```python
from abc import ABC, abstractmethod
from typing import Iterator, Any

class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def read(self) -> Iterator[Any]:
        pass
    
    @abstractmethod
    def close(self):
        pass

class DatabaseSource(DataSource):
    """数据库数据源"""
    
    def __init__(self, connection_string: str, query: str):
        self.connection_string = connection_string
        self.query = query
        self.conn = None
    
    def connect(self):
        import psycopg2
        self.conn = psycopg2.connect(self.connection_string)
    
    def read(self) -> Iterator[dict]:
        with self.conn.cursor() as cursor:
            cursor.execute(self.query)
            columns = [desc[0] for desc in cursor.description]
            
            for row in cursor:
                yield dict(zip(columns, row))
    
    def close(self):
        if self.conn:
            self.conn.close()

class FileSource(DataSource):
    """文件数据源"""
    
    def __init__(self, file_path: str, format: str = 'json'):
        self.file_path = file_path
        self.format = format
    
    def connect(self):
        pass
    
    def read(self) -> Iterator[dict]:
        if self.format == 'json':
            import json
            with open(self.file_path, 'r') as f:
                for line in f:
                    yield json.loads(line)
        elif self.format == 'csv':
            import csv
            with open(self.file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
    
    def close(self):
        pass
```

### 2.2 数据清洗层

```python
from typing import Callable, List
import logging

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.cleaning_steps: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_step(self, step: Callable, name: str = None):
        """添加清洗步骤"""
        step.__name__ = name or step.__name__
        self.cleaning_steps.append(step)
        return self
    
    def clean(self, data: Iterator[dict]) -> Iterator[dict]:
        """执行清洗流程"""
        stats = {'total': 0, 'passed': 0, 'failed': 0}
        
        for record in data:
            stats['total'] += 1
            
            try:
                cleaned_record = record
                for step in self.cleaning_steps:
                    cleaned_record = step(cleaned_record)
                    if cleaned_record is None:
                        break
                
                if cleaned_record is not None:
                    stats['passed'] += 1
                    yield cleaned_record
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                stats['failed'] += 1
                self.logger.error(f"清洗失败: {e}, 记录: {record}")
        
        self.logger.info(f"清洗统计: {stats}")

# 使用示例
cleaner = DataCleaner()
cleaner.add_step(remove_empty_records, "移除空记录")
cleaner.add_step(normalize_text, "文本规范化")
cleaner.add_step(filter_by_language, "语言过滤")
cleaner.add_step(deduplicate, "去重")
```

### 2.3 数据转换层

```python
class DataTransformer:
    """数据转换器"""
    
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, transformation: Callable):
        self.transformations.append(transformation)
        return self
    
    def transform(self, data: Iterator[dict]) -> Iterator[dict]:
        for record in data:
            transformed = record
            for transformation in self.transformations:
                transformed = transformation(transformed)
            yield transformed

# 常用转换函数
def add_timestamp(record: dict) -> dict:
    from datetime import datetime
    record['processed_at'] = datetime.utcnow().isoformat()
    return record

def flatten_nested(record: dict) -> dict:
    """展平嵌套结构"""
    flattened = {}
    
    def _flatten(obj, prefix=''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                _flatten(value, f"{prefix}{key}_")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                _flatten(value, f"{prefix}{i}_")
        else:
            flattened[prefix.rstrip('_')] = obj
    
    _flatten(record)
    return flattened

def encode_categorical(record: dict, field: str, mapping: dict) -> dict:
    """编码分类变量"""
    if field in record:
        record[field] = mapping.get(record[field], -1)
    return record
```

---

## 3. Pipeline 编排工具

### 3.1 Apache Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# 定义 DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['data-team@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    'ai_data_pipeline',
    default_args=default_args,
    description='AI 训练数据流水线',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:
    
    # 任务 1：数据采集
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_from_sources,
        op_kwargs={'sources': ['database', 'api', 'files']}
    )
    
    # 任务 2：数据清洗
    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_raw_data
    )
    
    # 任务 3：数据转换
    transform_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_for_training
    )
    
    # 任务 4：数据验证
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_dataset
    )
    
    # 任务 5：加载到训练环境
    load_task = PythonOperator(
        task_id='load_to_training',
        python_callable=upload_to_storage,
        op_kwargs={'destination': 's3://training-data/'}
    )
    
    # 定义依赖关系
    extract_task >> clean_task >> transform_task >> validate_task >> load_task
```

### 3.2 Prefect（现代替代方案）

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
import requests

@task(retries=3, retry_delay_seconds=5)
def fetch_api_data(url: str) -> dict:
    """从 API 获取数据"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@task(cache_key_fn=task_input_hash)
def process_data(raw_data: dict) -> list[dict]:
    """处理数据（带缓存）"""
    processed = []
    for item in raw_data['items']:
        processed.append({
            'id': item['id'],
            'text': item['content'],
            'timestamp': item['created_at']
        })
    return processed

@task
def save_to_database(data: list[dict], table: str):
    """保存到数据库"""
    import psycopg2
    
    conn = psycopg2.connect("postgresql://localhost/mydb")
    cursor = conn.cursor()
    
    for record in data:
        cursor.execute(
            f"INSERT INTO {table} (id, text, timestamp) VALUES (%s, %s, %s)",
            (record['id'], record['text'], record['timestamp'])
        )
    
    conn.commit()
    cursor.close()
    conn.close()

@flow(name="AI Data Pipeline", log_prints=True)
def ai_data_pipeline(source_url: str, target_table: str):
    """AI 数据流水线"""
    
    # 获取数据
    raw_data = fetch_api_data(source_url)
    
    # 处理数据
    processed = process_data(raw_data)
    
    # 保存数据
    save_to_database(processed, target_table)
    
    return f"成功处理 {len(processed)} 条记录"

# 运行
if __name__ == "__main__":
    ai_data_pipeline(
        source_url="https://api.example.com/data",
        target_table="training_data"
    )
```

### 3.3 轻量级方案：GitHub Actions + Makefile

```yaml
# .github/workflows/data-pipeline.yml
name: Data Pipeline

on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨 2 点
  workflow_dispatch:  # 支持手动触发

jobs:
  process-data:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data pipeline
      run: |
        make data-pipeline
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: processed-data
        path: data/processed/
```

```makefile
# Makefile
.PHONY: data-pipeline clean test

data-pipeline: data/raw data/processed
	python scripts/extract.py
	python scripts/clean.py
	python scripts/transform.py
	python scripts/validate.py
	python scripts/upload.py

data/raw:
	mkdir -p data/raw

data/processed:
	mkdir -p data/processed

clean:
	rm -rf data/processed/*
	rm -rf logs/*

test:
	pytest tests/
```

---

## 4. 数据质量监控

### 4.1 数据验证框架

```python
import great_expectations as gx
from great_expectations.core.expectation_suite import ExpectationSuite

def create_expectation_suite() -> ExpectationSuite:
    """创建数据验证规则"""
    
    context = gx.get_context()
    
    suite = context.add_expectation_suite(
        expectation_suite_name="training_data_validation"
    )
    
    # 规则 1：文本长度在合理范围
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="text_length",
            min_value=10,
            max_value=10000
        )
    )
    
    # 规则 2：语言字段不为空
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="language"
        )
    )
    
    # 规则 3：标签在预定义集合中
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="label",
            value_set=["positive", "negative", "neutral"]
        )
    )
    
    # 规则 4：无重复记录
    suite.add_expectation(
        gx.expectations.ExpectCompoundColumnsToBeUnique(
            column_list=["id"]
        )
    )
    
    return suite

def validate_dataframe(df, suite: ExpectationSuite) -> dict:
    """验证 DataFrame"""
    
    checkpoint = gx.checkpoint.SimpleCheckpoint(
        name="data_validation_checkpoint",
        data_context=gx.get_context(),
        validator=gx.validator.Validator(
            execution_engine="pandas",
            batches=[df],
            expectation_suite=suite
        )
    )
    
    results = checkpoint.run()
    
    return {
        'success': results.success,
        'statistics': results.statistics,
        'failed_expectations': [
            result for result in results.results
            if not result.success
        ]
    }
```

### 4.2 数据漂移检测

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping = None
) -> dict:
    """检测数据漂移"""
    
    report = Report(metrics=[DataDriftPreset()])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    # 获取漂移检测结果
    drift_results = report.as_dict()
    
    # 判断是否有显著漂移
    has_drift = any(
        metric['result']['drift_detected']
        for metric in drift_results['metrics']
        if 'result' in metric and 'drift_detected' in metric['result']
    )
    
    return {
        'has_drift': has_drift,
        'details': drift_results,
        'report_html': report.get_html()
    }
```

---

## 5. 数据版本管理

### 5.1 DVC (Data Version Control)

```bash
# 初始化 DVC
dvc init

# 跟踪数据集
dvc add data/training/dataset.jsonl

# 跟踪模型
dvc add models/best_model.pt

# 配置远程存储
dvc remote add -d myremote s3://mybucket/dvc-storage
dvc remote modify myremote region us-east-1

# 推送数据
dvc push

# 拉取特定版本
git checkout v1.0
dvc pull
```

```python
# Python API 使用 DVC
import dvc.api

# 读取特定版本的数据
with dvc.api.open(
    'data/training/dataset.jsonl',
    repo='https://github.com/user/repo',
    rev='v1.0'
) as fd:
    for line in fd:
        data = json.loads(line)
        process(data)
```

### 5.2 LakeFS（数据湖版本控制）

```python
import lakefs_client
from lakefs_client import models

# 创建分支（类似 Git 分支）
client = lakefs_client.LakeFSClient(
    configuration=lakefs_client.Configuration(
        host="http://localhost:8000",
        username="AKIAIOSFODNN7EXAMPLE",
        password="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )
)

# 创建实验分支
client.branches.create_branch(
    repository="ai-training-data",
    branch=models.BranchCreation(
        name="experiment-new-model",
        source="main"
    )
)

# 在分支上写入数据
# ...

# 合并回主分支
client.refs.merge_into_branch(
    repository="ai-training-data",
    source_ref="experiment-new-model",
    destination_branch="main"
)
```

---

## 6. 生产环境最佳实践

### 6.1 错误处理与重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class RobustPipeline:
    """健壮的 Pipeline 执行器"""
    
    def __init__(self):
        self.steps = []
        self.error_handlers = {}
    
    def add_step(self, step: Callable, name: str, 
                 error_handler: Callable = None):
        self.steps.append((name, step))
        if error_handler:
            self.error_handlers[name] = error_handler
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def execute_step(self, name: str, step: Callable, data: Any) -> Any:
        """执行单个步骤（带重试）"""
        logger.info(f"执行步骤: {name}")
        return step(data)
    
    def run(self, initial_data: Any) -> Any:
        """运行完整 Pipeline"""
        data = initial_data
        
        for name, step in self.steps:
            try:
                data = self.execute_step(name, step, data)
            except Exception as e:
                logger.error(f"步骤 {name} 失败: {e}")
                
                if name in self.error_handlers:
                    data = self.error_handlers[name](e, data)
                else:
                    raise
        
        return data
```

### 6.2 监控与告警

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# 定义监控指标
PIPELINE_RUNS = Counter('pipeline_runs_total', 'Pipeline 运行次数', ['status'])
PIPELINE_DURATION = Histogram('pipeline_duration_seconds', 'Pipeline 执行时间')
RECORDS_PROCESSED = Counter('records_processed_total', '处理记录数', ['stage'])
DATA_QUALITY_SCORE = Gauge('data_quality_score', '数据质量评分')

class MonitoredPipeline:
    """带监控的 Pipeline"""
    
    def run(self):
        start_time = time.time()
        
        try:
            # 执行 Pipeline
            result = self._execute()
            
            # 记录成功
            PIPELINE_RUNS.labels(status='success').inc()
            
            # 记录处理量
            RECORDS_PROCESSED.labels(stage='total').inc(result['count'])
            
            # 记录质量分数
            DATA_QUALITY_SCORE.set(result['quality_score'])
            
            return result
            
        except Exception as e:
            # 记录失败
            PIPELINE_RUNS.labels(status='failure').inc()
            raise
            
        finally:
            # 记录执行时间
            PIPELINE_DURATION.observe(time.time() - start_time)
```

---

## 💡 我的思考

1. **Pipeline 是 AI 系统的"血管"**：数据流的质量和稳定性直接决定模型效果，Pipeline 设计不能马虎。

2. **幂等性是关键**：Pipeline 步骤应该可以安全地重复执行，不会因为重复运行而产生错误结果。

3. **数据版本管理 = 模型版本管理**：没有数据版本管理，就无法复现模型训练。DVC 是轻量且有效的方案。

4. **监控比实现更重要**：Pipeline 运行失败时，快速定位问题比快速实现功能更有价值。

5. **从简单开始**：不要一开始就上 Airflow + Spark 的重型方案，Makefile + cron 可能足够用很久。

---

## 参考来源

- [Apache Airflow Documentation](https://airflow.apache.org/docs/) — 访问日期：2026-06-10
- [Prefect Documentation](https://docs.prefect.io/) — 访问日期：2026-06-10
- [Great Expectations Documentation](https://docs.greatexpectations.io/) — 访问日期：2026-06-10
- [DVC Documentation](https://dvc.org/doc) — 访问日期：2026-06-10
- [LakeFS Documentation](https://docs.lakefs.io/) — 访问日期：2026-06-10

---

*最后更新：2026-06-12*
