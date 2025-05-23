import numpy as np

# Load the .npz file
try:
    data_loaded = np.load('tang/tang.npz', allow_pickle=True)
except FileNotFoundError:
    print("Error: tang.npz not found. Make sure the file is in the correct directory.")
    exit()

# Retrieve data and ix2word
try:
    data_array = data_loaded['data']
    ix2word_numpy_array = data_loaded['ix2word']
    
    if ix2word_numpy_array.shape == ():
        ix2word = ix2word_numpy_array.item()
    else:
        print("Error: 'ix2word' is not a 0-d array and cannot be converted to a dictionary.")
        exit()
        
except KeyError as e:
    print(f"Error: Required key {e} not found in tang.npz.")
    exit()

# Define special tokens to exclude
# From previous subtask: 8292 is '</s>', 8291 is '<START>', 8290 is '<EOP>'
special_tokens_to_exclude = ['</s>', '<START>', '<EOP>']

all_poems_text = []

# Iterate through each poem in the data array
for poem_numerical in data_array:
    current_poem_text = []
    for word_id_val in poem_numerical:
        word_id = int(word_id_val) # Ensure it's a Python int for dict lookup
        
        word = ix2word.get(word_id)
        
        if word is None:
            # Handle cases where a word_id might not be in ix2word, though unlikely with this dataset
            # print(f"Warning: Word ID {word_id} not found in ix2word. Skipping.")
            continue
            
        if word not in special_tokens_to_exclude:
            current_poem_text.append(word)
            
    if current_poem_text: # Only add if the poem is not empty after stripping special tokens
        all_poems_text.append(current_poem_text)

# Print the first 3 poems to verify
print("First 3 processed poems for Gensim:")
for i, poem_text_list in enumerate(all_poems_text[:3]):
    print(f"Poem {i+1}: {' '.join(poem_text_list)}")

# Close the file
data_loaded.close()
