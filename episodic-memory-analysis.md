# Episodic Memory - Embedding Analysis & Improvement Brainstorm

## Current Implementation

### Embedding Model
- **Model**: `Xenova/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Max tokens**: 512 tokens (truncated to 2000 characters)
- **Framework**: @xenova/transformers (Transformers.js v2.17.2)
- **Runtime**: ONNX Runtime (CPU-only, Node.js)
- **Performance**: ~100-200 embeddings/sec on modern CPU

### Architecture
```
Input Text (2000 char limit)
    ↓
@xenova/transformers (CPU)
    ↓
all-MiniLM-L6-v2 (384-dim)
    ↓
SQLite + sqlite-vec (vector search)
    ↓
Cosine similarity search
```

### Database Layer
- **Primary DB**: better-sqlite3 (SQLite)
- **Vector extension**: sqlite-vec v0.1.7-alpha.2
- **Vector table**: FLOAT[384] with MATCH operator for cosine similarity
- **Search modes**: vector (semantic), text (LIKE), both (hybrid)

### Key Strengths
1. **Zero dependencies** - Fully local, no API calls for embeddings
2. **Privacy** - All data stays on local machine
3. **Simplicity** - Single SQLite file, easy backup/migration
4. **Fast setup** - No GPU required, works everywhere
5. **Portable** - Pure Node.js, runs on any platform

### Current Limitations
1. **Small model** - all-MiniLM-L6-v2 is 6 layers, basic performance
2. **CPU-only** - Can't leverage GPU for faster embedding generation
3. **Low dimensions** - 384-dim may miss nuances vs 768/1024-dim models
4. **Token limit** - 512 tokens may truncate longer conversations
5. **Older model** - Released 2020, newer models significantly better

## Benchmark: Model Quality Comparison

### MTEB Leaderboard (Massive Text Embedding Benchmark)

| Model | Dims | Avg Score | Speed | Size | Notes |
|-------|------|-----------|-------|------|-------|
| **Current: all-MiniLM-L6-v2** | 384 | 56.3 | Fast | 80MB | Baseline |
| all-mpnet-base-v2 | 768 | 57.8 | Medium | 420MB | Better quality |
| gte-small | 384 | 61.5 | Fast | 67MB | Same dims, much better |
| e5-small-v2 | 384 | 62.8 | Fast | 134MB | SOTA for 384-dim |
| e5-base-v2 | 768 | 65.0 | Medium | 438MB | Best balanced option |
| bge-small-en-v1.5 | 384 | 62.2 | Fast | 133MB | Great for code/tech |
| nomic-embed-text-v1.5 | 768 | 62.4 | Medium | 548MB | Prefix-based search |
| **gte-Qwen2-1.5B-instruct** | 1536 | 70.2 | Slow | 3GB | SOTA but huge |

**Recommendation tier:**
- **Quick win**: gte-small (384-dim, +5.2 points, same speed)
- **Best balanced**: e5-base-v2 (768-dim, +8.7 points, 2x slower but worth it)
- **Code-focused**: bge-small-en-v1.5 (384-dim, optimized for code/technical text)

## GPU Acceleration Options

### Option 1: Transformers.js with ONNX GPU (Simplest)
**Pros:**
- Minimal code changes
- Keep current architecture
- Works with existing models

**Cons:**
- ONNX GPU support in Node.js is limited/experimental
- @xenova/transformers doesn't officially support GPU in v2.x
- Complex setup (CUDA, cuDNN, onnxruntime-node-gpu)

**Verdict**: Not recommended - too much hassle for marginal gains

### Option 2: HuggingFace TEI (Text Embeddings Inference) - GPU Native
**Pros:**
- Production-grade GPU inference server
- 3-10x faster than CPU for batch embedding
- Supports all modern embedding models
- Same models as Transformers.js (drop-in upgrade)
- Can reuse serena's claude-context infrastructure!

**Cons:**
- Requires Docker + NVIDIA GPU
- External service dependency
- Network overhead for single embeddings

**Implementation:**
```typescript
// Instead of @xenova/transformers
import fetch from 'node-fetch';

async function generateEmbedding(text: string): Promise<number[]> {
  const response = await fetch('http://localhost:8080/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: text })
  });
  const result = await response.json();
  return result[0]; // TEI returns array of embeddings
}
```

**Verdict**: Best option if you want GPU acceleration + can run Docker

### Option 3: Integrate with Serena's Claude-Context
**Pros:**
- Reuse existing GPU infrastructure (you already have this running!)
- Milvus vector DB (production-grade, handles millions of vectors)
- Better than SQLite for large-scale vector search
- Unified semantic search across code + conversations

**Cons:**
- Tighter coupling to serena ecosystem
- More complex setup/configuration
- Milvus is heavier than SQLite

**Architecture:**
```
Episodic Memory (conversations)
    ↓
Claude-Context TEI (GPU embeddings)
    ↓
Milvus (vector search)
    ↑
Serena (code)
```

**Verdict**: Most powerful, but only if you want unified search infrastructure

### Option 4: Python Bridge + sentence-transformers (GPU)
**Pros:**
- Direct access to sentence-transformers library (most models available)
- Easy GPU support via PyTorch
- Great for experimentation

**Cons:**
- Adds Python dependency to Node.js project
- IPC overhead (spawn Python process per embedding)
- Complexity in deployment

**Verdict**: Good for prototyping, not production

## Recommended Upgrade Paths

### Path A: Easy Win (No GPU)
**Goal**: Better embeddings, same architecture

**Changes:**
1. Upgrade model to `Xenova/gte-small` or `Xenova/bge-small-en-v1.5`
2. Keep @xenova/transformers + SQLite
3. Change 1 line of code

**Impact:**
- +5-6 MTEB points (10% better quality)
- Same speed, same dependencies
- Zero architectural changes

**Code:**
```typescript
// embeddings.ts line 10
embeddingPipeline = await pipeline(
  'feature-extraction',
  'Xenova/gte-small'  // or 'Xenova/bge-small-en-v1.5'
);
```

**Effort**: 5 minutes
**Risk**: Very low (same API, same dims)

---

### Path B: Best Quality (No GPU)
**Goal**: Best embeddings without GPU infrastructure

**Changes:**
1. Upgrade model to `Xenova/e5-base-v2` (768-dim)
2. Keep @xenova/transformers + SQLite
3. Update vector table schema (384 → 768)
4. Re-index all conversations

**Impact:**
- +8.7 MTEB points (15% better quality)
- 2x slower embedding generation (still acceptable for background indexing)
- Higher precision for nuanced queries

**Code:**
```typescript
// embeddings.ts
embeddingPipeline = await pipeline(
  'feature-extraction',
  'Xenova/e5-base-v2'
);

// db.ts line 99
CREATE VIRTUAL TABLE IF NOT EXISTS vec_exchanges USING vec0(
  id TEXT PRIMARY KEY,
  embedding FLOAT[768]  // Changed from 384
)
```

**Migration:**
```bash
# Backup existing index
cp ~/.episodic-memory/conversations.db ~/.episodic-memory/conversations.db.backup

# Re-index with new model
episodic-memory index --cleanup --force
```

**Effort**: 30 minutes + re-indexing time
**Risk**: Low (well-tested models, clear migration path)

---

### Path C: GPU Acceleration (TEI)
**Goal**: Fast GPU embedding + better models

**Changes:**
1. Run HuggingFace TEI in Docker (similar to claude-context setup)
2. Replace @xenova/transformers with HTTP client
3. Support batch embedding for faster indexing
4. Optional: upgrade to 768-dim model

**Architecture:**
```typescript
// New: tei-embeddings.ts
import fetch from 'node-fetch';

let teiBaseUrl = process.env.TEI_BASE_URL || 'http://localhost:8080';

export async function generateEmbedding(text: string): Promise<number[]> {
  const response = await fetch(`${teiBaseUrl}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: text })
  });
  const data = await response.json();
  return data[0];
}

// Batch support for faster indexing
export async function generateEmbeddingBatch(texts: string[]): Promise<number[][]> {
  const response = await fetch(`${teiBaseUrl}/embed`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ inputs: texts })
  });
  return await response.json();
}
```

**Docker setup:**
```bash
# Run TEI with GPU (reuse serena's approach)
docker run -d \
  --gpus all \
  -p 8081:80 \
  -v $HOME/.cache/huggingface:/data \
  ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-base-en-v1.5 \
  --max-batch-tokens 16384
```

**Impact:**
- 3-10x faster embedding generation (GPU)
- Batch processing (100s of embeddings/sec)
- Better models (bge, gte, e5 all supported)
- Falls back to CPU if TEI unavailable

**Effort**: 2-3 hours (Docker setup, code changes, testing)
**Risk**: Medium (new dependency, requires GPU)

---

### Path D: Unified Search (Serena Integration)
**Goal**: Single semantic search infrastructure for code + conversations

**Changes:**
1. Expose claude-context's embedding endpoint
2. Replace SQLite vector table with Milvus collection
3. Add conversation collection to existing Milvus instance
4. Use serena's MCP tools for unified search

**Benefits:**
- Search code AND conversations in one query
- Production-grade vector DB (Milvus)
- Already have the infrastructure running
- Unified embedding model across use cases

**Architecture:**
```
┌─────────────────────────────────────────┐
│ Claude-Context (TEI + Milvus)          │
├─────────────────────────────────────────┤
│ Collections:                            │
│  - code_embeddings (existing)           │
│  - conversation_embeddings (new)        │
└─────────────────────────────────────────┘
         ↑                    ↑
         │                    │
    ┌────┴────┐         ┌─────┴──────┐
    │ Serena  │         │ Episodic   │
    │  (MCP)  │         │  Memory    │
    └─────────┘         └────────────┘
```

**Code:**
```typescript
// Use serena's claude-context client
import { ClaudeContextClient } from '@serena/claude-context-client';

const client = new ClaudeContextClient({
  baseUrl: 'http://localhost:8080'
});

export async function generateEmbedding(text: string): Promise<number[]> {
  return await client.embed(text);
}

// Use Milvus for vector search
export async function searchConversations(
  query: string,
  options: SearchOptions
): Promise<SearchResult[]> {
  const embedding = await generateEmbedding(query);

  return await client.search({
    collection: 'conversation_embeddings',
    vector: embedding,
    limit: options.limit || 10,
    filter: buildMilvusFilter(options) // after/before date filtering
  });
}
```

**Impact:**
- GPU-accelerated embeddings
- Scalable vector search (millions of conversations)
- Unified search ("find code examples from conversations about X")
- Better search quality (Milvus > SQLite for vectors)

**Effort**: 4-6 hours (Milvus schema, client integration, migration)
**Risk**: Medium-high (architectural change, dependency on serena)

---

## Comparison Matrix

| Path | Quality Gain | Speed Gain | GPU Required | Effort | Risk | Cost |
|------|--------------|------------|--------------|--------|------|------|
| **A: Easy Win** | +10% | 0% | No | 5 min | Very Low | Free |
| **B: Best Quality** | +15% | -50% | No | 30 min | Low | Free |
| **C: GPU TEI** | +15% | +300-1000% | Yes | 2-3 hrs | Medium | GPU |
| **D: Serena Unified** | +15% | +300-1000% | Yes | 4-6 hrs | Med-High | GPU |

## My Recommendation

**Start with Path A (Easy Win), then Path C (GPU TEI) if needed.**

**Why:**
1. **Path A** is 5 minutes, zero risk, meaningful improvement
2. **Path C** gives you 90% of Path D benefits without tight coupling
3. You already know how to run TEI (from claude-context experience)
4. Keep SQLite for simplicity - it's fine for conversation search
5. Reserve Path D for when you actually need unified code+conversation search

**Implementation order:**
```bash
# Phase 1: Quick win (today)
# Update model to gte-small or bge-small-en-v1.5
# Test search quality improvement

# Phase 2: GPU acceleration (when indexing becomes slow)
# Run TEI in Docker
# Add batch embedding support
# Fallback to Transformers.js if TEI unavailable

# Phase 3: Consider unified search (future)
# Only if you find yourself wanting to search code + conversations together
# Only if SQLite performance becomes an issue (unlikely until 100k+ conversations)
```

## Testing Strategy

### Quality Testing
```typescript
// test/embeddings-comparison.test.ts
const testQueries = [
  "How did we handle React Router authentication?",
  "What was the bug with async tests?",
  "Explain the TDD approach we used",
  "Find conversations about git workflows"
];

// Compare models
for (const model of ['all-MiniLM-L6-v2', 'gte-small', 'e5-base-v2']) {
  const results = await searchWithModel(model, testQueries);
  // Check if better models return more relevant results
}
```

### Performance Testing
```typescript
// Measure embedding generation time
const texts = generateTestConversations(100);

const start = Date.now();
for (const text of texts) {
  await generateEmbedding(text);
}
const elapsed = Date.now() - start;

console.log(`${texts.length} embeddings in ${elapsed}ms`);
console.log(`${(texts.length / (elapsed / 1000)).toFixed(1)} embeddings/sec`);
```

## Next Steps

1. **Read this analysis** - discuss which path makes sense for your use case
2. **Path A prototype** - I can make the 1-line change and test quality
3. **Path C prototype** - If you want GPU, I can set up TEI + batch support
4. **Path D exploration** - If unified search interests you, we can design the integration

What direction interests you most?
