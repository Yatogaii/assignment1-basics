import os
from typing import Tuple, List
import regex as re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab_size = len(vocab)
        self.merges = merges
        self.vocab = vocab
        
    def train(self, data):
        # Placeholder for training logic
        pass
    
    def encode(self, text):
        # Placeholder for tokenization logic
        return text.split()  # Simple whitespace tokenizer for illustration
    
    def decode(self, tokens):
        # Placeholder for detokenization logic
        return ' '.join(tokens)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  
def process_pretoken(chunk:str) -> dict[bytes, int]:
    pretoken_freqs = {}
    result = re.findall(PAT, chunk)
        
    for each_pretoken in result:
        bts = each_pretoken.encode("utf-8")
        pretoken_freqs[bts] = pretoken_freqs.get(bts, 0) + 1
        
    return pretoken_freqs
    
def pre_tokenization(input_path, special_tokens, desired_num_chunks) -> dict[bytes, int]:
    # read file
    f=  open(input_path, 'rb')
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    f.seek(0)
    
    special_tokens_bts = [each.encode("utf-8") for each in special_tokens]
    
    chunk_size = file_size // desired_num_chunks
    # 代表每个 Chunk 的边界
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        f.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = f.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            pattern = b'|'.join(re.escape(token) for token in special_tokens_bts)
            match = re.search(pattern, mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.start()
                break
                
            #if initial_position > 3*mini_chunk_size+chunk_boundaries[bi]:
            #    chunk_boundaries[bi] = initial_position
            #    break
            
            initial_position += mini_chunk_size

    # real pre-token process after chunk split.
    from multiprocessing import Pool, Process  
    num_process = 6


    chunks = []
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks.append(chunk)
        
    all_pretoken_freqs = {}
    with Pool(num_process) as pool:
        list_of_counters = pool.map(process_pretoken, chunks)

        for counter in list_of_counters:
            for key, value in counter.items():
                all_pretoken_freqs[key] = all_pretoken_freqs.get(key, 0) + value
    f.close()

    return all_pretoken_freqs

def replace_pair(words: tuple, pair: tuple, token: int) -> tuple:
    '''
    Replace all occurrences of a pair in words with a new token.
    
    Args:
        words: input tuple, our target
        pair: the pair to be replaced
        token: the new token id to replace the pair
    
    Returns:
        new_tuple: the tuple after replacement
    '''
    # 构造 new_tuple (左到右、不重叠地替换)
    new_tuple = []
    i = 0
    
    while i < len(words):
        if i + 1 < len(words) and (words[i], words[i+1]) == pair:
            new_tuple.append(token)
            i += 2  # 跳过两个元素
        else:
            new_tuple.append(words[i])
            i += 1
    
    return tuple(new_tuple)
    
def train_BPETokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab :dict[int, bytes] = {}
    merges:list[tuple[bytes, bytes]] = []
    
    all_token_freqs = pre_tokenization(input_path, special_tokens, 6)
    
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    
    # reversed_vocab: string -> token
    reversed_vocab: dict[bytes, int] = {}
    index = -1
    
    # init vocab
    for i in range(256):
        vocab[i] = i.to_bytes(1, 'little')
        reversed_vocab[i.to_bytes(1, 'little')] = i
        
    index = 255
    
    # handle the special token to vocab
    special_token_sets= set(special_tokens)
    for i in range(len(special_tokens)):
        index += 1
        bts = special_tokens[i].encode('utf-8')
        vocab[index] = bts
        reversed_vocab[bts] = index
        
    assert(len(vocab) == 256 + len(special_tokens))
    
    # caculate the pre-tokens map
    int_token_freqs = {}
    for each in all_token_freqs.keys():
        key_int_list = []
        for per in each:
            per_bts = per.to_bytes(1, "little")
            key_int_list.append(reversed_vocab[per_bts])
        key_int = tuple(key_int_list)
        int_token_freqs[key_int] = all_token_freqs[each]
    
    # get raw pair-count
    # for update pair_cnt incrementally, we should get all pair's appear location in int_token_freqs
    pair_cnt = {}
    pair_words_maps: dict[tuple, set[tuple]] = {}
    word_pairs_map: dict[tuple, set[tuple]] = {}
    for pretoken, value in int_token_freqs.items():
        for i in range(len(pretoken)-1):
            pair = (pretoken[i], pretoken[i+1])
            pair_cnt[pair] = pair_cnt.get(pair, 0) + value
            
            if pair not in pair_words_maps:
                pair_words_maps[pair] = set()
            
            # key->index, to quickly get the reference of int_token_freqs
            pair_words_maps[pair].add(pretoken)
    for k,v in pair_words_maps.items():
        for each in v:
            if each not in word_pairs_map:
                word_pairs_map[each] = set()
            word_pairs_map[each].add(k)
            

    # Main training loop, exit when vocab size reaches the target size.
    while len(vocab) < vocab_size:
        # STEP1: get the max pair, best_pair: tuple(int)
        # Debug: 查看前5个最大值
        top5_pairs = sorted(pair_cnt.items(), key=lambda x: x[1], reverse=True)[:5]
        if len(top5_pairs) >= 2 and top5_pairs[0][1] == top5_pairs[1][1]:
            print(f"Tie detected: {top5_pairs[0][0]} vs {top5_pairs[1][0]}")
        
        # 选择频率最高的 pair，如果频率相同则选字典序最小的
        # 排序规则：1) 频率降序（越大越好） 2) pair 升序（字典序最小）
        sorted_pairs = sorted(pair_cnt.items(), key=lambda x: (-x[1], x[0]))
        best_pair = sorted_pairs[0][0]   
        
        # STEP2: update vocab
        index += 1
        bts = []
        for each in best_pair:
            bts.append(vocab[each])
        vocab[index] = b''.join(bts)
        reversed_vocab[best_pair] = index
        
        # STEP3: update the pair_words_maps and int_token_freqs and pair_cnt
        # UPDATE LOGIC:
        # we use an snapshot if pair_words_maps to represent all pretokens that contains best_pair.
        # then, we pop the best_pair for pair_words_map and pair_cnt.
        # We construct an new token by replace the best_pair to new token.
        # And we update pair_cnt and pair_words_maps by it's prev_pair and next_pair
        # Note, the snapshot contains raw pretoken that contains the best_pair.
        snapshot = set(pair_words_maps[best_pair])
        
        pair_cnt.pop(best_pair)
        pair_words_maps.pop(best_pair)        
        
        for old_word in snapshot:
            old_value = int_token_freqs[old_word]
            new_words = replace_pair(old_word, best_pair, index)
            int_token_freqs[new_words] = int_token_freqs.get(new_words, 0) + old_value 
            int_token_freqs.pop(old_word)
            
            # Calculate all pairs in old_word and new_words
            old_pairs = set()
            old_pair_counts = {}  # 同时记录每个 pair 的出现次数
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i+1])
                old_pairs.add(pair)
                old_pair_counts[pair] = old_pair_counts.get(pair, 0) + 1
            
            new_pairs = set()
            new_pair_counts = {}  # 同时记录每个 pair 的出现次数
            for i in range(len(new_words) - 1):
                pair = (new_words[i], new_words[i+1])
                new_pairs.add(pair)
                new_pair_counts[pair] = new_pair_counts.get(pair, 0) + 1
            
            # Remove pairs that only exist in old_word (excluding best_pair which was already popped)
            removed_pairs = old_pairs - new_pairs
            for pair in removed_pairs:
                if pair == best_pair:
                    continue
                pair_cnt[pair] -= old_pair_counts[pair] * old_value  # 用实际出现次数
                if pair_cnt[pair] <= 0:
                    pair_cnt.pop(pair, None)
                if pair in pair_words_maps:
                    pair_words_maps[pair].discard(old_word)
                    if not pair_words_maps[pair]:
                        pair_words_maps.pop(pair, None)
            
            # Add pairs that only exist in new_words
            added_pairs = new_pairs - old_pairs
            for pair in added_pairs:
                pair_cnt[pair] = pair_cnt.get(pair, 0) + new_pair_counts[pair] * old_value  # 用实际出现次数
                if pair not in pair_words_maps:
                    pair_words_maps[pair] = set()
                pair_words_maps[pair].add(new_words)
            
            # Update pair_words_maps for pairs that exist in both (change reference from old_word to new_words)
            common_pairs = old_pairs & new_pairs
            for pair in common_pairs:
                delta = new_pair_counts[pair] - old_pair_counts[pair]
                if delta != 0:
                    pair_cnt[pair] = pair_cnt.get(pair, 0) + delta * old_value
                    if pair_cnt[pair] <= 0:
                        pair_cnt.pop(pair, None)
                
                if pair in pair_words_maps:
                    pair_words_maps[pair].discard(old_word)
                    pair_words_maps[pair].add(new_words) 

        #best_pair_bts = tuple([each.to_bytes(1,"little") for each in best_pair])
        new_merge = []
        for i in [0,1]:
            new_merge.append(vocab[best_pair[i]])
        merges.append(tuple(new_merge))
        print(f'vocab size = {len(vocab)}, data size = {len(all_token_freqs)}')
    
    return (vocab, merges)

if __name__ == '__main__':
    # Test case for train_BPETokenizer function
    print("=" * 60)
    print("=== Test train_BPETokenizer ===")
    print("=" * 60)
    
    # Create a more comprehensive test file with repeated patterns
    test_file = "test_bpe_input.txt"
    test_text = """The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
Hello world! Hello everyone! Hello there!
Python is great. Python is amazing. Python programming is fun.
Machine learning and deep learning are fascinating.
Natural language processing with transformers.
Tokenization is an important preprocessing step.
Byte pair encoding helps with subword tokenization.
This is a test. This is only a test. Testing testing 123.
<|endoftext|>
The end of the first document.
<|endoftext|>
Starting a new document here.
Repeated words: test test test example example example.
More repeated patterns: the the the and and and.
<|endoftext|>
"""
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    print(f"\nInput text length: {len(test_text)} characters")
    print(f"Input text preview (first 200 chars):")
    print(f"  {test_text[:200]}...")
    print(f"\nTarget vocab size: 300")
    print(f"Special tokens: ['<|endoftext|>']\n")
    
    print("-" * 60)
    print("Training BPE Tokenizer...")
    print("-" * 60)
    
    # Train the tokenizer
    vocab, merges = train_BPETokenizer(
        input_path=test_file,
        vocab_size=300,
        special_tokens=['<|endoftext|>']
    )
    
    print("\n" + "=" * 60)
    print("=== Training Results ===")
    print("=" * 60)
    
    print(f"\n✓ Final vocab size: {len(vocab)}")
    print(f"✓ Number of merges: {len(merges)}")
    print(f"✓ Initial vocab (bytes + special): 256 + 1 = 257")
    print(f"✓ Learned merges: {len(merges)}")
    
    # Show first 10 merges
    print(f"\n--- First 10 Merges ---")
    for i, (left, right) in enumerate(merges[:10], 1):
        merged = left + right
        try:
            merged_str = merged.decode('utf-8', errors='ignore')
            print(f"  {i:2d}. {left!r:20s} + {right!r:20s} -> {merged!r:25s} ('{merged_str}')")
        except:
            print(f"  {i:2d}. {left!r:20s} + {right!r:20s} -> {merged!r:25s}")
    
    # Show last 10 merges
    if len(merges) > 10:
        print(f"\n--- Last 10 Merges ---")
        for i, (left, right) in enumerate(merges[-10:], len(merges)-9):
            merged = left + right
            try:
                merged_str = merged.decode('utf-8', errors='ignore')
                print(f"  {i:2d}. {left!r:20s} + {right!r:20s} -> {merged!r:25s} ('{merged_str}')")
            except:
                print(f"  {i:2d}. {left!r:20s} + {right!r:20s} -> {merged!r:25s}")
    
    # Show learned vocab entries
    print(f"\n--- Learned Vocabulary Tokens (beyond base 257) ---")
    print(f"Showing tokens {257} to {min(277, len(vocab)-1)}:")
    for token_id in range(257, min(277, len(vocab))):
        token_bytes = vocab[token_id]
        try:
            token_str = token_bytes.decode('utf-8', errors='ignore')
            print(f"  Token {token_id:3d}: {token_bytes!r:25s} -> '{token_str}'")
        except:
            print(f"  Token {token_id:3d}: {token_bytes!r:25s}")
    
    # Check for common patterns
    print(f"\n--- Analysis of Common Patterns ---")
    common_patterns = [b'th', b'he', b'in', b'er', b'an', b'ed', b'or', b'll', 
                       b'ing', b'the', b' the', b' and', b'est', b'The']
    found_patterns = []
    for pattern in common_patterns:
        for token_id, token_bytes in vocab.items():
            if token_id >= 257 and token_bytes == pattern:
                found_patterns.append((pattern, token_id))
                break
    
    if found_patterns:
        print("Found common English patterns in vocabulary:")
        for pattern, token_id in found_patterns:
            print(f"  '{pattern.decode('utf-8', errors='ignore')}' -> Token {token_id}")
    else:
        print("Note: Common patterns might be part of larger merged tokens")
    
    # Expected behavior summary
    print(f"\n" + "=" * 60)
    print("=== Expected Behavior Summary ===")
    print("=" * 60)
    print(f"✓ Vocab size should be exactly 400")
    print(f"✓ Should have {400 - 257} merges (400 - 256 bytes - 1 special token)")
    print(f"✓ Common English pairs should be merged early:")
    print(f"  - Space + common words like ' the', ' and'")
    print(f"  - Common letter pairs like 'th', 'he', 'in', 'er'")
    print(f"  - Common endings like 'ing', 'ed', 'ly'")
    print(f"✓ Frequently repeated words like 'test', 'The', 'Python' should form tokens")
    print(f"✓ The special token '<|endoftext|>' should be in vocab at position 256")
    
    # Verify special token
    print(f"\n--- Special Token Verification ---")
    special_token_bytes = '<|endoftext|>'.encode('utf-8')
    if 256 in vocab and vocab[256] == special_token_bytes:
        print(f"✓ Special token correctly placed at index 256: {vocab[256]}")
    else:
        print(f"✗ Special token not found at expected position 256")
    
    # Expected merges (manually calculated)
    print(f"\n" + "=" * 60)
    print("=== Expected First Merges (Manual Calculation) ===")
    print("=" * 60)
    print("\nBased on the test text, the most frequent byte pairs should be:")
    
    expected_first_merges = [
        (b'e', b' ', "Space after 'e' appears very frequently"),
        (b' ', b't', "Space before 't' (common: 'the', 'test', 'to', etc.)"),
        (b't', b'h', "'th' is very common in English"),
        (b'h', b'e', "'he' forms 'the' and appears in 'there', 'Hello', etc."),
        (b'i', b'n', "'in' is a common word and suffix"),
        (b'e', b'r', "'er' appears in many words"),
        (b'o', b'n', "'on' appears in 'Tokenization', 'encoding', etc."),
        (b'a', b'n', "'an' appears in 'and', 'language', etc."),
        (b'i', b's', "'is' is a common word"),
        (b' ', b'a', "Space before 'a' (common: 'and', 'an', 'are', etc.)"),
    ]
    
    print("\nTop 10 expected merges (approximate, order may vary by 1-2):")
    for i, (left, right, reason) in enumerate(expected_first_merges, 1):
        merged = left + right
        try:
            merged_str = merged.decode('utf-8')
            print(f"  {i:2d}. {left!r} + {right!r} -> {merged!r} ('{merged_str}') - {reason}")
        except:
            print(f"  {i:2d}. {left!r} + {right!r} -> {merged!r} - {reason}")
    
    print("\nNote: Exact order depends on precise frequency counts from the text.")
    print("After initial byte-pair merges, higher-level merges will form:")
    print("  - Common words: 'the', 'is', 'test', 'and'")
    print("  - Common prefixes: 'The ', 'Th', 'ing'")
    print("  - Frequent patterns from repeated sentences")
    
    # Compare actual vs expected
    print(f"\n--- Verification: Compare First 10 Actual vs Expected ---")
    matches = 0
    for i in range(min(10, len(merges))):
        actual_merge = merges[i]
        actual_str = (actual_merge[0] + actual_merge[1]).decode('utf-8', errors='ignore')
        
        # Check if this merge appears in our expected list
        is_expected = any(
            (left, right) == actual_merge or (right, left) == actual_merge
            for left, right, _ in expected_first_merges
        )
        
        status = "✓" if is_expected else "?"
        print(f"  {status} Merge {i+1}: {actual_merge[0]!r} + {actual_merge[1]!r} -> '{actual_str}'")
        if is_expected:
            matches += 1
    
    print(f"\nMatches with expected patterns: {matches}/10")
    if matches >= 7:
        print("✓ Good! Most early merges match expected patterns.")
    elif matches >= 4:
        print("⚠ Partial match. Some differences might be due to exact frequency counts.")
    else:
        print("✗ Warning: Merges differ significantly from expected patterns.")
    
    # Cleanup
    import os
    os.remove(test_file)
    print(f"\n✓ Test file cleaned up successfully")
    print("=" * 60)
    