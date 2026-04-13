# Plan: SFTDataset Parquet and Conversations Support

---

## Objective

Modify `SFTLowLevelDataset` in Megatron-LM to:
1. Auto-detect file format (JSONL vs Parquet) via HuggingFace `datasets.load_dataset()`
2. Support both `messages` and `conversations` column names (for org's nemotron schema compatibility)

---

## Files to Modify

| File | Change |
|------|--------|
| `megatron/training/datasets/sft_dataset.py` | SFTLowLevelDataset class |

---

## Changes

### 1. `SFTLowLevelDataset.__init__` (line ~39-46)

**What:** Auto-detect file format instead of hardcoding "json"
**Why:** Allow loading both JSONL and Parquet files
**Source:** HuggingFace datasets library documentation

```python
# FROM:
self.dataset = load_dataset("json", data_files=dataset_path, split="all")

# TO:
self.dataset = load_dataset(data_files=dataset_path, split="all")
```

### 2. `SFTLowLevelDataset.__getitem__` (line ~52)

**What:** Add fallback from `messages` to `conversations` column
**Why:** Support org's nemotron schema which uses `conversations`
**Source:** `finetune.py:274-276` already uses this pattern

```python
# FROM:
return self.dataset[idx]["messages"]

# TO:
item = self.dataset[idx]
return item.get("messages", item.get("conversations"))
```

---

## Verification

### Create Test Parquet File

**What:** Create a minimal parquet file for testing
**Location:** `tests/unit_tests/test_data/sft_test_conversations.parquet`

**Schema:**
```python
{
    "conversations": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}
```

### Add Unit Test

**What:** Add test to verify parquet loading with `conversations` column
**Location:** `tests/unit_tests/test_datasets/test_sft_dataset.py`

```python
def test_sft_parquet_with_conversations():
    """Test SFTLowLevelDataset loads parquet with conversations column."""
    dataset_path = "tests/unit_tests/test_data/sft_test_conversations.parquet"
    dataset = SFTLowLevelDataset(dataset_path)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample[0]["role"] == "system"
    assert sample[1]["role"] == "user"
```

---

## Testing Checklist

| Test | Description |
|------|-------------|
| JSONL with `messages` | Existing behavior - verify no regression |
| JSONL with `conversations` | New fallback path |
| Parquet with `messages` | New format support |
| Parquet with `conversations` | New format + fallback combined |

---

## Architecture Context

```
SFTDataset (lines 55-215)
    │
    ├── Uses SFTLowLevelDataset internally
    ├── build_low_level_dataset() → SFTLowLevelDataset(dataset_path)
    │
    └── __getitem__() → tokenization, masking, packing happens HERE
                         (NO changes needed - works with any message list)

SFTLowLevelDataset (lines 21-52) ← CHANGES HERE
    │
    ├── __init__() → load_dataset() ← AUTO-DETECT FORMAT
    │
    └── __getitem__() → returns raw messages ← FALLBACK messages/conversations
```

---

## Success Criteria

1. `load_dataset(data_files=path)` works with both `.jsonl` and `.parquet` extensions
2. `__getitem__` returns data from `messages` if present, else `conversations`
3. All existing SFT tests pass (no regression)
4. New test parquet file loads correctly with `conversations` column

---

## Dependencies

- `datasets` library (already required for SFTDataset)
- `pyarrow` or `pandas` (for creating test parquet file)

---

## Out of Scope

- Modifying `MockSFTLowLevelDataset` - generates synthetic data, not file-based
- Dataset conversion tools
- Supporting other column names beyond `messages` and `conversations`
