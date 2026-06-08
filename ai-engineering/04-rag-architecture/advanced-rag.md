# 高级 RAG 技术

> **一句话定位**：掌握重排序、查询扩展、多路召回等高级技术，构建生产级 RAG 系统。
>
> #status/canonical #topic/rag #topic/advanced #year/2026

---

## 1. 查询优化技术

### 1.1 查询重写

**问题**：用户查询可能模糊、不完整或与文档表述不一致。

**解决方案**：使用 LLM 重写查询。

```python
class QueryRewriter:
    def __init__(self, model):
        self.model = model
    
    def rewrite(self, query, strategy="expansion"):
        """
        重写查询以提高检索效果
        """
        if strategy == "expansion":
            return self._expand_query(query)
        elif strategy == "decomposition":
            return self._decompose_query(query)
        elif strategy == "hyde":
            return self._hyde_query(query)
        else:
            return query
    
    def _expand_query(self, query):
        """
        查询扩展：添加同义词和相关术语
        """
        prompt = f"""
        为以下查询生成 3-5 个语义相似的扩展查询，
        帮助检索更多相关文档。
        
        原始查询：{query}
        
        扩展查询（每行一个）：
        """
        
        expansions = self.model.generate(prompt).strip().split('\n')
        return [query] + [e.strip('- ') for e in expansions if e.strip()]
    
    def _decompose_query(self, query):
        """
        查询分解：将复杂查询拆分为子查询
        """
        prompt = f"""
        将以下复杂查询分解为 2-4 个简单的子查询，
        每个子查询应该独立可回答。
        
        复杂查询：{query}
        
        子查询：
        1.
        """
        
        sub_queries = self.model.generate(prompt).strip().split('\n')
        return [q.strip('123456789. ') for q in sub_queries if q.strip()]
    
    def _hyde_query(self, query):
        """
        HyDE: Hypothetical Document Embeddings
        生成假设文档，用假设文档做检索
        """
        prompt = f"""
        为以下查询生成一个假设的文档段落（100字左右），
        这个段落应该包含查询的答案。
        
        查询：{query}
        
        假设文档：
        """
        
        hypothetical_doc = self.model.generate(prompt)
        return hypothetical_doc
```

### 1.2 查询理解

```python
class QueryUnderstanding:
    def __init__(self, model):
        self.model = model
    
    def analyze(self, query):
        """
        分析查询意图和实体
        """
        prompt = f"""
        分析以下查询，提取关键信息：
        
        查询：{query}
        
        请输出：
        1. 查询意图（信息查找/比较/操作/其他）
        2. 关键实体
        3. 时间范围（如有）
        4. 需要的文档类型
        
        以 JSON 格式输出：
        """
        
        analysis = self.model.generate(prompt)
        return json.loads(analysis)
    
    def route_query(self, query, analyzers):
        """
        根据查询类型路由到不同的处理流程
        """
        analysis = self.analyze(query)
        intent = analysis['intent']
        
        # 路由到不同的检索策略
        if intent == 'comparison':
            return analyzers['comparison'].handle(query)
        elif intent == 'temporal':
            return analyzers['temporal'].handle(query)
        else:
            return analyzers['default'].handle(query)
```

---

## 2. 多路召回

### 2.1 并行检索

```python
class MultiRetriever:
    def __init__(self):
        self.retrievers = {
            'keyword': BM25Retriever(),
            'semantic': DenseRetriever(),
            'graph': GraphRetriever(),
        }
    
    def retrieve(self, query, top_k=10):
        """
        多路召回：并行检索，融合结果
        """
        import concurrent.futures
        
        # 并行检索
        all_results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                name: executor.submit(r.search, query, top_k*2)
                for name, r in self.retrievers.items()
            }
            
            for name, future in futures.items():
                all_results[name] = future.result()
        
        # 融合结果
        fused_results = self._fuse_results(all_results, top_k)
        return fused_results
    
    def _fuse_results(self, all_results, top_k):
        """
        使用 RRF (Reciprocal Rank Fusion) 融合结果
        """
        k = 60  # RRF 常数
        scores = {}
        
        for retriever_name, results in all_results.items():
            for rank, (doc, score) in enumerate(results, 1):
                doc_id = doc.id
                if doc_id not in scores:
                    scores[doc_id] = {'doc': doc, 'score': 0}
                
                # RRF 公式
                scores[doc_id]['score'] += 1 / (k + rank)
        
        # 排序并返回 Top-K
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [(r['doc'], r['score']) for r in sorted_results[:top_k]]
```

### 2.2 分层检索

```python
class HierarchicalRetriever:
    def __init__(self):
        self.summary_index = SummaryIndex()  # 高层摘要
        self.detail_index = DetailIndex()    # 详细内容
    
    def retrieve(self, query, top_k=5):
        """
        分层检索：先粗后细
        """
        # 第一层：在摘要中检索
        coarse_results = self.summary_index.search(query, top_k=top_k*3)
        
        # 获取相关文档的详细内容
        detailed_docs = []
        for doc, score in coarse_results:
            detail = self.detail_index.get_document(doc.id)
            detailed_docs.append(detail)
        
        # 第二层：在详细内容中精确检索
        fine_results = self.detail_index.search_in_docs(
            query, 
            documents=detailed_docs,
            top_k=top_k
        )
        
        return fine_results
```

---

## 3. 上下文处理

### 3.1 上下文压缩

```python
class ContextCompressor:
    def __init__(self, model):
        self.model = model
    
    def compress(self, contexts, max_tokens=2000):
        """
        压缩上下文以适应模型窗口
        """
        # 1. 提取关键句子
        key_sentences = self._extract_key_sentences(contexts)
        
        # 2. 去除冗余
        unique_sentences = self._remove_redundancy(key_sentences)
        
        # 3. 按重要性排序
        ranked_sentences = self._rank_by_importance(unique_sentences)
        
        # 4. 截断到最大长度
        compressed = self._truncate(ranked_sentences, max_tokens)
        
        return compressed
    
    def _extract_key_sentences(self, contexts):
        """提取关键句子"""
        key_sentences = []
        for ctx in contexts:
            sentences = sent_tokenize(ctx)
            # 使用 TF-IDF 或模型评分选择关键句子
            for sent in sentences:
                score = self._score_sentence(sent)
                key_sentences.append((sent, score))
        return key_sentences
    
    def _remove_redundancy(self, sentences, threshold=0.85):
        """去除语义冗余的句子"""
        unique = []
        for sent, score in sentences:
            is_redundant = False
            for existing, _ in unique:
                sim = semantic_similarity(sent, existing)
                if sim > threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                unique.append((sent, score))
        return unique
```

### 3.2 上下文重排序

```python
class ContextReranker:
    def __init__(self, cross_encoder):
        self.cross_encoder = cross_encoder
    
    def rerank(self, query, contexts, top_k=5):
        """
        使用 Cross-Encoder 精确重排序
        """
        # 构建查询-文档对
        pairs = [(query, ctx) for ctx in contexts]
        
        # 计算相关性分数
        scores = self.cross_encoder.predict(pairs)
        
        # 排序
        ranked = sorted(
            zip(contexts, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:top_k]
```

---

## 4. 生成优化

### 4.1 引用生成

```python
class CitationGenerator:
    def __init__(self, model):
        self.model = model
    
    def generate_with_citations(self, query, contexts):
        """
        生成带引用的答案
        """
        # 为每个上下文添加编号
        numbered_contexts = []
        for i, ctx in enumerate(contexts, 1):
            numbered_contexts.append(f"[{i}] {ctx}")
        
        prompt = f"""
        基于以下文档回答问题。在答案中引用相关文档的编号 [1], [2] 等。
        
        文档：
        {'\n'.join(numbered_contexts)}
        
        问题：{query}
        
        要求：
        1. 每个事实都必须有引用
        2. 如果没有足够信息，说明"无法确定"
        3. 不要编造信息
        
        答案：
        """
        
        answer = self.model.generate(prompt)
        
        # 验证引用
        validated_answer = self._validate_citations(answer, contexts)
        return validated_answer
    
    def _validate_citations(self, answer, contexts):
        """验证引用是否准确"""
        # 提取引用
        citations = re.findall(r'\[(\d+)\]', answer)
        
        # 检查每个引用
        for cite in citations:
            idx = int(cite) - 1
            if idx >= len(contexts):
                # 引用不存在，标记问题
                answer = answer.replace(f'[{cite}]', f'[INVALID:{cite}]')
        
        return answer
```

### 4.2 多文档融合

```python
class MultiDocumentFusion:
    def __init__(self, model):
        self.model = model
    
    def fuse(self, query, documents):
        """
        融合多个文档的信息生成答案
        """
        # 1. 提取每个文档的关键信息
        key_points = []
        for doc in documents:
            points = self._extract_key_points(query, doc)
            key_points.extend(points)
        
        # 2. 去重和聚类
        unique_points = self._cluster_points(key_points)
        
        # 3. 组织成连贯答案
        answer = self._organize_answer(query, unique_points)
        
        return answer
    
    def _extract_key_points(self, query, document):
        """提取与查询相关的关键信息点"""
        prompt = f"""
        从以下文档中提取与问题相关的关键信息点（每点一行）：
        
        问题：{query}
        文档：{document}
        
        关键信息点：
        - 
        """
        
        points = self.model.generate(prompt).strip().split('\n')
        return [p.strip('- ') for p in points if p.strip()]
    
    def _cluster_points(self, points):
        """聚类相似的信息点"""
        # 使用嵌入聚类
        embeddings = model.encode(points)
        clusters = cluster_embeddings(embeddings)
        
        # 从每个聚类选择最具代表性的点
        unique_points = []
        for cluster in clusters:
            center = find_center(cluster)
            unique_points.append(points[center])
        
        return unique_points
```

---

## 5. 高级架构模式

### 5.1 Self-RAG

```python
class SelfRAG:
    def __init__(self, retriever, generator, critic):
        self.retriever = retriever
        self.generator = generator
        self.critic = critic
    
    def generate(self, query, max_iterations=3):
        """
        Self-RAG: 自适应检索和生成
        """
        context = []
        
        for i in range(max_iterations):
            # 生成初步答案
            draft = self.generator.generate(query, context)
            
            # 评估是否需要更多检索
            need_retrieval = self.critic.evaluate(query, draft, context)
            
            if not need_retrieval:
                return draft
            
            # 检索更多信息
            new_docs = self.retriever.search(query, exclude=context)
            context.extend(new_docs)
        
        # 最终生成
        return self.generator.generate(query, context)
    
    def _reflect(self, query, answer, context):
        """
        反思：检查答案是否完整准确
        """
        prompt = f"""
        检查以下答案是否完整准确地回答了问题。
        
        问题：{query}
        答案：{answer}
        
        检查项：
        1. 答案是否包含所有必要信息？
        2. 是否有遗漏的方面？
        3. 是否需要更多背景信息？
        
        如果需要更多信息，请说明需要检索什么。
        """
        
        reflection = self.critic.generate(prompt)
        return reflection
```

### 5.2 Corrective RAG

```python
class CorrectiveRAG:
    def __init__(self, retriever, generator, evaluator):
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
    
    def generate(self, query):
        """
        Corrective RAG: 检索-评估-修正循环
        """
        # 1. 初始检索
        docs = self.retriever.search(query)
        
        # 2. 评估检索质量
        retrieval_quality = self.evaluator.evaluate_retrieval(query, docs)
        
        if retrieval_quality < 0.5:
            # 检索质量低，尝试改写查询
            query = self._rewrite_query(query)
            docs = self.retriever.search(query)
        
        # 3. 生成答案
        answer = self.generator.generate(query, docs)
        
        # 4. 评估答案质量
        answer_quality = self.evaluator.evaluate_answer(query, answer, docs)
        
        if answer_quality < 0.5:
            # 答案质量低，补充检索
            additional_docs = self.retriever.search(answer, exclude=docs)
            docs.extend(additional_docs)
            answer = self.generator.generate(query, docs)
        
        return answer
```

---

## 6. 性能优化

### 6.1 缓存策略

```python
class RAGCache:
    def __init__(self, embedding_cache, result_cache):
        self.embedding_cache = embedding_cache
        self.result_cache = result_cache
    
    def get_or_compute_embedding(self, text):
        """缓存 Embedding"""
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = model.encode(text)
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def get_or_retrieve(self, query, retriever):
        """缓存检索结果"""
        cache_key = hash(query)
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        results = retriever.search(query)
        self.result_cache[cache_key] = results
        return results
```

### 6.2 异步处理

```python
import asyncio

class AsyncRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    async def generate(self, query):
        """异步 RAG 流程"""
        # 异步检索
        docs = await self.retriever.asearch(query)
        
        # 异步生成
        answer = await self.generator.agenerate(query, docs)
        
        return answer
    
    async def batch_generate(self, queries):
        """批量异步处理"""
        tasks = [self.generate(q) for q in queries]
        return await asyncio.gather(*tasks)
```

---

## 参考资源

- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) - Asai et al., 2023
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) - Yan et al., 2024
- [Reciprocal Rank Fusion outperforms Condorcet](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Cormack et al., 2009
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) - Gao et al., 2022
- [Advanced RAG Techniques](https://www.pinecone.io/learn/series/rag/)
