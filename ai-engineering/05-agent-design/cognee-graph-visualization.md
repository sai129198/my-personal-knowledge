# Cognee 图谱结构与前端渲染方案

> **一句话定位**：理解 cognee 知识图谱的内部结构，掌握数据导出方式和前端可视化渲染的完整方案。
>
> #status/canonical #topic/memory #topic/knowledge-graph #topic/frontend-visualization #tool/cognee #year/2026

---

## 1. 图谱数据结构

cognee 的知识图谱遵循**属性图模型**，核心是两种元素：实体节点和关系边。

### 1.1 实体节点 (Nodes / Entities)

```python
# 假设输入: "Alice is a software engineer in Paris. She uses Python."
# cognee 自动提取:
节点1: { name: "Alice",           type: "Person" }
节点2: { name: "Paris",           type: "City" }
节点3: { name: "software engineer", type: "Occupation" }
节点4: { name: "Python",          type: "ProgrammingLanguage" }
```

**节点属性结构**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 唯一标识 |
| `name` | str | 实体名称 |
| `type` | str | 语义类型（Person / City / Date / Organization...） |
| `description` | str | LLM 生成的描述 |
| `chunk_id` | UUID | 来源 chunk（可追溯） |
| `document_id` | UUID | 来源文档 |
| `metadata` | dict | 自定义元数据 |
| `embedding` | vector | 向量嵌入（存在向量库中） |

### 1.2 关系边 (Relationships / Edges)

```python
# 从 "Alice is a software engineer in Paris" 自动提取:
边1: (Alice) -[works_as]->      (software engineer)
边2: (Alice) -[lives_in]->      (Paris)
边3: (Alice) -[uses]->          (Python)
```

**边属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `source_node_id` | UUID | 源节点 |
| `target_node_id` | UUID | 目标节点 |
| `relationship_type` | str | 关系类型（works_as / lives_in / uses...） |
| `description` | str | 自然语言描述 |
| `weight` | float | 权重（受 feedback 和 frequency 影响） |
| `chunk_id` | UUID | 证据来源 chunk |

---

## 2. 图谱的层次结构

cognee 的图谱不是扁平的一层，而是**四层递进**：

```
┌─────────────────────────────────────────────────────┐
│    第3层: 衍生知识（Derived Knowledge）               │
│    improve() 生成：推断出的新实体/关系                  │
│    例如: (Alice)→(France) 由 lives_in Paris           │
│          + "Paris 在法国" 推断得出                     │
├─────────────────────────────────────────────────────┤
│    第2层: 提取的实体 & 关系（Extracted Graph）         │
│    cognify 生成：来自文本的实体和关系                   │
│    Person → lives_in → City                         │
├─────────────────────────────────────────────────────┤
│    第1层: 文本块层（Chunk Layer）                      │
│    chunking 生成：chunk_1, chunk_2, ...              │
│    每个 chunk 有 embedding 用于语义检索                │
├─────────────────────────────────────────────────────┤
│    第0层: 文档层（Document Layer）                     │
│    ingestion 生成：document_id, metadata              │
│    记录来源、格式、创建时间等                          │
└─────────────────────────────────────────────────────┘
```

**跨层关联**：
- Document → Chunk：1:N
- Chunk → Entity：N:M（一个 chunk 含多个实体，一个实体可能跨多个 chunk）
- Entity → Entity：通过 Relationship 边连接
- 衍生层在原始实体关系基础上生成新连接

---

## 3. 数据导出方式

### 3.1 `export()` — 导出图谱文件

```python
import cognee

# COGX：Cognee 自有交换格式（保留完整图谱结构+元数据+溯源）
await cognee.export("my_dataset", format="cogx")

# GraphML：通用图交换格式，可导入 Gephi、Cytoscape 等
await cognee.export("my_dataset", format="graphml")

# Cypher：Neo4j 查询语言，可直接在 Neo4j 中执行还原
await cognee.export("my_dataset", format="cypher")

# JSON：通用数据格式
await cognee.export("my_dataset", format="json")
```

### 3.2 `get_schema_inventory()` — 图谱结构概览

```python
from cognee import get_schema_inventory

inventory = await get_schema_inventory()
# → 返回各节点类型的数量和关系统计
# {
#   "Person": {"count": 42, "sample_names": ["Alice", "Bob", ...], "relationships": [...]},
#   "City":   {"count": 18, "sample_names": ["Paris", "NYC", ...], "relationships": [...]},
#   ...
# }
```

前端可以用这个做**图例**和**类型筛选**。

### 3.3 导出 provenance 图（所有权关系图）

```python
from cognee.modules.storage.visualization import get_memory_provenance_graph

nodes, edges = await get_memory_provenance_graph()
# nodes: [(id, label, type), ...]
# edges: [(source, target, label), ...]
```

### 3.4 `recall()` 获取子图数据

```python
# 以查询为中心，获取相关子图
results = await cognee.recall(
    "Tell me about Alice",
    datasets=["team_knowledge"],
    query_type="GRAPH_COMPLETION",
    top_k=20,
)

# 从 result.raw 中提取子图节点和边用于渲染
for r in results:
    if r.source == "graph":
        subgraph = r.raw  # 包含该查询命中的节点和关系
```

---

## 4. 前端渲染方案

### 4.1 Cognee 官方内置（零代码）

```python
from cognee import visualize_memory_provenance

# 一键生成交互式 HTML，浏览器打开即可
await visualize_memory_provenance()
```

适用场景：内部调试、快速验证。

---

### 4.2 vis-network / vis.js — 推荐起步方案

**优点**：5 分钟出效果，自动力导向布局，缩放拖拽，click/hover 交互。

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div id="graph" style="width:100%; height:600px; border:1px solid #ccc;"></div>
  <script>
    // ===== 从后端 API 获取数据后 =====
    const nodes = new vis.DataSet([
      { id: 1,  label: "Alice",            group: "Person",    title: "Software engineer based in Paris" },
      { id: 2,  label: "Paris",            group: "City",      title: "Capital of France" },
      { id: 3,  label: "Software Engineer", group: "Occupation" },
      { id: 4,  label: "Python",           group: "Technology" },
    ]);
    const edges = new vis.DataSet([
      { from: 1, to: 2, label: "lives_in" },
      { from: 1, to: 3, label: "works_as" },
      { from: 1, to: 4, label: "uses" },
    ]);

    new vis.Network(
      document.getElementById("graph"),
      { nodes, edges },
      {
        groups: {
          Person:     { color: { background: "#4fc3f7", border: "#0288d1" } },
          City:       { color: { background: "#a5d6a7", border: "#388e3c" } },
          Occupation: { color: { background: "#ffcc80", border: "#f57c00" } },
          Technology: { color: { background: "#ce93d8", border: "#7b1fa2" } },
        },
        edges: {
          arrows: "to",
          smooth: { type: "curvedCW", roundness: 0.15 },
          font: { size: 9, color: "#666" },
        },
        physics: {
          solver: "forceAtlas2Based",
          forceAtlas2Based: { gravitationalConstant: -26, centralGravity: 0.005 },
        },
        interaction: {
          hover: true,
          tooltipDelay: 200,
          navigationButtons: true,
          keyboard: true,
        },
        layout: { improvedLayout: true },
      }
    );
  </script>
</body>
</html>
```

---

### 4.3 Cytoscape.js — 大规模图谱首选

**优点**：支持 1000+ 节点不卡，丰富的布局算法，插件生态完善。

```javascript
import cytoscape from "cytoscape";
// 可选插件
import coseBilkent from "cytoscape-cose-bilkent";
import dagre from "cytoscape-dagre";
import klay from "cytoscape-klay";

cytoscape.use(coseBilkent);

const cy = cytoscape({
  container: document.getElementById("cy"),

  elements: [
    { data: { id: "alice", label: "Alice" }, classes: "person" },
    { data: { id: "paris", label: "Paris" }, classes: "city" },
    { data: { id: "e1", source: "alice", target: "paris", label: "lives_in" } },
    { data: { id: "e2", source: "alice", target: "se", label: "works_as" } },
  ],

  style: [
    { selector: "node", style: {
        "label": "data(label)",
        "text-valign": "center",
        "text-halign": "center",
        "font-size": "11px",
    }},
    { selector: ".person", style: { "background-color": "#4fc3f7" } },
    { selector: ".city",   style: { "background-color": "#a5d6a7" } },
    { selector: "edge", style: {
        "width": 1.5,
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#999",
        "line-color": "#999",
        "curve-style": "bezier",
        "label": "data(label)",
        "font-size": "9px",
        "color": "#666",
    }},
  ],

  // CoSE Bilkent: 大图效果好，速度也快
  layout: { name: "cose-bilkent", nodeRepulsion: 4500 },

  // 交互
  wheelSensitivity: 0.3,
  minZoom: 0.1,
  maxZoom: 3,
});

// 点击节点展开详情
cy.on("tap", "node", (evt) => {
  const node = evt.target;
  console.log("Clicked:", node.data("label"));
  // 高亮邻居
  cy.elements().removeClass("highlighted");
  node.addClass("highlighted");
  node.neighborhood().addClass("highlighted");
});

// 按类型筛选
document.getElementById("filter-person").onclick = () => {
  cy.nodes(".person").toggleClass("hidden");
};
```

**布局选择指南**：

| 布局名 | 适用场景 | 节点数上限 |
|--------|---------|-----------|
| `cose` | 通用，复合弹簧嵌入 | ~500 |
| `cose-bilkent` | 大规模图谱 | ~5000 |
| `breadthfirst` | 层级/树形结构 | ~200 |
| `klay` | 流程图/DAG | ~300 |
| `dagre` | 有向层次图 | ~200 |
| `concentric` | 辐射状 | ~200 |

---

### 4.4 D3-force — 最大自定义空间

**优点**：完全控制每个像素。**缺点**：需要写更多代码。

```javascript
import * as d3 from "d3";

const width = 800, height = 600;

const svg = d3.select("#graph")
  .append("svg").attr("width", width).attr("height", height);

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(edges).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(30));

const link = svg.append("g")
  .selectAll("line").data(edges).join("line")
  .attr("stroke", "#999").attr("stroke-width", 1.5)
  .attr("marker-end", "url(#arrow)");

const node = svg.append("g")
  .selectAll("circle").data(nodes).join("circle")
  .attr("r", 12)
  .attr("fill", d => colorMap[d.type] || "#ccc")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
  );

const label = svg.append("g")
  .selectAll("text").data(nodes).join("text")
  .text(d => d.name).attr("font-size", 10).attr("dx", 15).attr("dy", 4);

simulation.on("tick", () => {
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  label.attr("x", d => d.x).attr("y", d => d.y);
});
```

---

## 5. React 组件集成示例

### 5.1 React + vis-network

```tsx
import { useEffect, useRef } from "react";
import { Network, DataSet } from "vis-network/standalone";

interface GraphData {
  nodes: { id: string; name: string; type: string; description?: string }[];
  edges: { source: string; target: string; label: string }[];
}

const TYPE_COLORS: Record<string, string> = {
  Person: "#4fc3f7", City: "#a5d6a7", Organization: "#ffcc80",
  Date: "#ef9a9a", Technology: "#ce93d8",
};

export function KnowledgeGraph({ dataset }: { dataset: string }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`/api/cognee/graph/${dataset}`)
      .then(r => r.json())
      .then((data: GraphData) => {
        const nodes = new DataSet(data.nodes.map(n => ({
          id: n.id, label: n.name,
          group: n.type,
          title: n.description,  // hover 显示
        })));
        const edges = new DataSet(data.edges.map(e => ({
          from: e.source, to: e.target, label: e.label,
        })));

        const groups = Object.fromEntries(
          Object.entries(TYPE_COLORS).map(([type, color]) => [
            type, { color: { background: color, border: "#333" } }
          ])
        );

        new Network(containerRef.current!, { nodes, edges }, {
          groups,
          edges: { arrows: "to", smooth: { type: "curvedCW" }, font: { size: 9 } },
          physics: { solver: "forceAtlas2Based" },
          interaction: { hover: true, navigationButtons: true },
        });
      });
  }, [dataset]);

  return <div ref={containerRef} style={{ width: "100%", height: "600px" }} />;
}
```

### 5.2 React + Cytoscape.js

```tsx
import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";

export function CytoscapeGraph({ elements }: { elements: cytoscape.ElementDefinition[] }) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const cy = cytoscape({
      container: containerRef.current!,
      elements,
      style: [
        { selector: "node", style: { "label": "data(label)", "font-size": "10px" } },
        { selector: "edge", style: { "target-arrow-shape": "triangle", "curve-style": "bezier" } },
      ],
      layout: { name: "cose-bilkent" },
    });
    return () => cy.destroy();
  }, [elements]);

  return <div ref={containerRef} style={{ width: "100%", height: "600px" }} />;
}
```

---

## 6. 后端 API 设计建议

```python
from fastapi import FastAPI
import cognee

app = FastAPI()

@app.get("/api/cognee/graph/{dataset}")
async def get_graph_data(dataset: str):
    """返回图谱节点和边，前端直接灌入可视化引擎"""
    # 导出为 GraphML，在服务端解析后转为前端友好的 JSON
    graphml = await cognee.export(dataset, format="graphml")
    nodes, edges = parse_graphml(graphml)  # 自行实现解析
    return {"nodes": nodes, "edges": edges}

@app.get("/api/cognee/schema/{dataset}")
async def get_schema(dataset: str):
    """返回节点类型统计，前端做图例和筛选"""
    return await cognee.get_schema_inventory()

@app.get("/api/cognee/subgraph")
async def get_subgraph(query: str, dataset: str = "main_dataset"):
    """查询返回相关子图"""
    results = await cognee.recall(
        query, datasets=[dataset],
        query_type="GRAPH_COMPLETION",
        top_k=20,
    )
    subgraph = extract_subgraph(results)  # 从 raw 中提取
    return subgraph

@app.get("/api/cognee/provenance")
async def get_provenance():
    """所有权关系图"""
    nodes, edges = await cognee.get_memory_provenance_graph()
    return {"nodes": nodes, "edges": edges}
```

---

## 7. 方案选择速查

| 场景 | 推荐方案 |
|------|---------|
| 快速验证、内部调试 | `visualize_memory_provenance()` 一键生成 HTML |
| 小规模图谱（<500 节点） | vis-network，5 行 JS 出效果 |
| 中大规模（500-5000 节点） | Cytoscape.js + CoSE-Bilkent 布局 |
| 需要完全自定义 UI | D3-force + 手写渲染 |
| 前后端分离架构 | REST API（导出/检索） + 前端图谱库 |
| React 项目 | vis-network 或 Cytoscape.js 封装为 React 组件 |

---

## 相关文档

- [Cognee 深度解析](./cognee-deep-dive.md)
- [Cognee 接口 API 参考](./cognee-api-reference.md)
- [Agent 记忆系统设计](./agent-memory.md)
- [官方 Graph Visualization 指南](https://docs.cognee.ai/guides/graph-visualization)
