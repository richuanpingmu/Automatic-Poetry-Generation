import numpy as np

# Load the .npz file
try:
    data_loaded = np.load('tang/tang.npz', allow_pickle=True)
except FileNotFoundError:
    print("Error: tang.npz not found. Make sure the file is in the correct directory.")
    exit()

# Print the keys
print("Keys in the .npz file:", list(data_loaded.keys()))

# Print the shape of the 'data' array
if 'data' in data_loaded:
    print("Shape of 'data' array:", data_loaded['data'].shape)
else:
    print("'data' array not found in the .npz file.")

# Print the type of 'ix2word' and 'word2ix'
if 'ix2word' in data_loaded:
    print("Type of 'ix2word':", type(data_loaded['ix2word']))
else:
    print("'ix2word' not found in the .npz file.")

if 'word2ix' in data_loaded:
    print("Type of 'word2ix':", type(data_loaded['word2ix']))
else:
    print("'word2ix' not found in the .npz file.")

# Retrieve ix2word and word2ix
ix2word = None
word2ix = None

if 'ix2word' in data_loaded and data_loaded['ix2word'].shape == ():
    ix2word = data_loaded['ix2word'].item()
    print("\\nFirst 10 items in ix2word:")
    for i, (k, v) in enumerate(ix2word.items()):
        if i < 10:
            print(f"  {k}: {v}")
        else:
            break
else:
    print("'ix2word' is not a 0-d array or not found.")

if 'word2ix' in data_loaded and data_loaded['word2ix'].shape == ():
    word2ix = data_loaded['word2ix'].item()
    print("\\nFirst 10 items in word2ix:")
    for i, (k, v) in enumerate(word2ix.items()):
        if i < 10:
            print(f"  {k}: {v}")
        else:
            break
else:
    print("'word2ix' is not a 0-d array or not found.")

if ix2word:
    print(f"\\nToken for 8291: {ix2word.get(8291)}")
    print(f"Token for 8292: {ix2word.get(8292)}")
    
# Print the first poem
if 'data' in data_loaded and ix2word is not None:
    first_poem_numerical = data_loaded['data'][0]
    print("\\nFirst poem (numerical):", first_poem_numerical)

    print("\\nFirst poem (textual):")
    poem_text = []
    started = False
    for i, word_ix_val in enumerate(first_poem_numerical):
        if i >= 125: # Max length including padding
            break
        
        # Ensure word_ix_val is a Python int if it's a numpy type
        word_ix = int(word_ix_val) 
        
        word = ix2word.get(word_ix, f"<UNK_IDX_{word_ix}>")

        if word == '</s>': # Token 8292 is </s>
            if started: # If we have already started collecting words, then </s> means end of poem.
                break
            else: # If we haven't started, this is leading padding, so skip.
                continue
        else:
            started = True
            # Token 8291 is <START>. We can choose to include or exclude it.
            # Let's exclude it for cleaner output as per typical conventions.
            if word == '<START>':
                continue # Skip adding <START> to the poem text
            poem_text.append(word)
            
    print(" ".join(poem_text))

else:
    if 'data' not in data_loaded:
        print("'data' array not found, cannot print first poem.")
    if ix2word is None:
        print("'ix2word' not available, cannot convert poem to text.")

# Close the file
data_loaded.close()
