class G2PData (data.Dataset):
    def __init__(self, graphemes, phonemes):
        self.graphemes = graphemes
        self.phonemes = phonemes

    def __len__(self):
        return len(self.graphemes)

    def __getitem__(self, index):
        graphemes = self.graphemes[index]
        phonemes = self.phonemes[index]

        #Fetches encoded versions
        grapheme_vector = data_encoder(graphemes, 1)
        phoneme_vector = data_encoder(phonemes, 0)

        #Omits </s> character
        decoder_inputs = phoneme_vector[:-1]
        phoneme_vector = phoneme_vector[1:]

        #Used for padding purposes
        g_vec_len = len(grapheme_vector) 
        p_vec_len = len(phoneme_vector)
        
        return grapheme_vector, phoneme_vector, decoder_inputs, g_vec_len, p_vec_len

def data_decoder(sequence, isWord):
    """Converts index sequence back into corresponding letter tokens"""
    if isWord: tokenizer = cfg.index2graph
    else: tokenizer = cfg.index2phone
    converted_sequence = []
    
    for i in sequence:
        if tokenizer[i] == "</s>": break
        a = tokenizer[i]
        converted_sequence.append(a)
    return converted_sequence

def data_encoder(seq, isWord):
    # Automatically encoders sequence with graph2index if words
    tokenized_seq = []
    if isWord: 
        seq = [*seq] + ['</s>']
        seq = [i for i in seq if i!=" "]
        for i in seq:
            a = cfg.graph2index[i]
            tokenized_seq.append(a)
    #Else simply add end of sequence token to to phoneme sequences
    else:
        a = '<s> ' + str(seq) +' </s>'
        seq = a.split(" ")
        ans = ""
        for i in seq:
            if i== 'o': i="O"
            elif i== 'h': i="h_"
            a = cfg.phone2index[i]
            tokenized_seq.append(a)

    #Tokenize sequence
    return tokenized_seq

