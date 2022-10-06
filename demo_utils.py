import torch
import re
from torch.utils import data

class Config:
        seed = '5'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 200
        batch_size = 128
        hidden_dim = 256
        embed_dim = 256
        dropout = 0.1
        dec_max_len = 30
        MAX_LENGTH = 20
        teacher_forcing_ratio = 0.5
        n_layers = 2
        lr = 0.001
        graphemes = ['<pad>', '<unk>', '</s>', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'ä', 'è', 'é', 'ê', 'ë', 'í', 'ï', 'ó', 'ô', 'ö', 'ú', 'û']
        phonemes = ['<pad>', '<unk>', '<s>', '</s>', '2:', '9', '9y', '@', '@i', '@u', 'A:', 'E', 'N', 'O', 'Of', 'S', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h_', 'i', 'i@', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'u@', 'v', 'w', 'x', 'y', 'z', '{']   
        graph2index = {g: idx for idx, g in enumerate(graphemes)}
        index2graph = {idx: g for idx, g in enumerate(graphemes)}

        phone2index = {p: idx for idx, p in enumerate(phonemes)}
        index2phone = {idx: p for idx, p in enumerate(phonemes)}
        g_vocab_size = len(graphemes)
        p_vocab_size = len(phonemes)

cfg = Config()

def split_nums_letters(seq):
    nums = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','30','40','50','60','70','80','90','100']
    num_words = ['nul', 'een','twee', 'drie','vier','vyf', 'ses','sewe', 'agt', 'nege','tien','elf','twaalf','dertien','veertien','vyftien','sestien','sewentien', 'agtien', 'negentien', 
        'twintig','dertig', 'veertig', 'vyftig', 'sestig', 'sewentig', 'tagtig', 'negentig', 'honderd']
    
    num_tens = ['10', '20','30','40','50','60','70','80','90']
    num_words_tens = ['tien', 'twintig','dertig', 'veertig', 'vyftig', 'sestig', 'sewentig', 'tagtig', 'negentig']
    nums_dict_combined = dict(zip(nums,num_words))
    nums_tens_dict = dict(zip(num_tens, num_words_tens))

    def num_to_word(h,t,u, flag, max_len, curr_str_len):
        ans = ""
        h_added = 0
        t_u_combined = 0
        req_suffix = 0
        end = ['biljoen', 'miljoen', 'duisend']
        #print(f"h{h}, t{t}, u{u}")
        if h:
            if h!="0":
                ans = ans + nums_dict_combined[h] + " honderd"
                h_added = 1
                req_suffix = 1
        if h=='0' and (t!="0" or u!="0") and curr_str_len: 
            ans = ans + "en "

        if (t+u):
            if (t+u)!="00" and nums_dict_combined.get(t+u):
                    if h_added: ans = ans +" en "+ nums_dict_combined[t+u]
                    else: ans = ans + nums_dict_combined[t+u]
                    t_u_combined = 1
                    req_suffix = 1

        if t_u_combined == 0:
            if t!="0" and u!="0" and t_u_combined == 0:
                    if h_added: ans = ans +" "+ nums_dict_combined[u] + " en " +  num_words_tens[int(t)-1]
                    else: ans = ans + nums_dict_combined[u] + " en " +  num_words_tens[int(t)-1]
                    req_suffix = 1

            elif t!="0" and t_u_combined == 0:
                    if h_added: ans = ans +" en "+ nums_tens_dict[int(t)]
                    else: ans = ans + nums_dict_combined[t]
                    req_suffix = 1

            elif u!="0" and t_u_combined == 0:
                if h_added: ans = ans +" en "+ nums_dict_combined[u]
                else: ans = ans + nums_dict_combined[u]
                req_suffix = 1

        if flag!=(max_len-1) and req_suffix and len(ans): ans = ans +" " + end[flag]

        return ans



    def convert_num_seq(num):

        a = int(num)
        max_len = 4


        num_word_equiv = ""
        tu_both = 0
        t_added = 0
        h_added = 0
        tt_th_both = 0
        tt_added = 0
        ht_added = 0

        bu = str(int(a/1000000000)-10*int(a/10000000000))
        bt = str(int(a/10000000000)-10*int(a/100000000000))
        bh = str(int(a/100000000000))
        
        mu = str(int(a/1000000)-10*int(a/10000000))
        mt = str(int(a/10000000)-10*int(a/100000000))
        mh = str(int((a%1000000000)/100000000))

        uth = str(int(a/1000)-10*int(a/10000))
        tth = str(int(a/10000)-10*int(a/100000))
        hth = str(int((a%1000000)/100000))

        u =  str(int(a%10))
        t =  str(int((a%100)/10))
        h =  str(int((a%1000)/100))
        #t_u = str(a%100)

        digits = [bh, bt, bu, mh, mt, mu, hth, tth, uth, h, t ,u]

        for i in range(max_len):
            flag = i #Use to add hondered or miljoen
            new_str = num_to_word(digits[i*3], digits[i*3+1], digits[(i*3)+2], flag, max_len, len(num_word_equiv))
            #if len(num_word_equiv) and len(new_str):num_word_equiv = num_word_equiv + " en " +  new_str
            if len(new_str):
                if len(num_word_equiv): num_word_equiv +=  (" " + new_str)
                else: num_word_equiv +=  (new_str)

            
        return num_word_equiv

    c = re.findall(r'[A-Za-z]+|\d+', seq) #Source: https://stackoverflow.com/questions/28290492/python-splitting-numbers-and-letters-into-sub-strings-with-regular-expression
    ans = []
    for j,i in enumerate(c):
        if i.isdigit():
            ans.append(convert_num_seq(i))
        else: ans.append(i)

    return ans

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

def generate_pd(model, iter, device):

    graphemes, phonemes = [],[]
    with torch.no_grad():
        for i,batch in enumerate(iter):
            grapheme_vector, phoneme_vector, decoder_inputs, g_vec_len, p_vec_len = batch
            grapheme_vector = grapheme_vector.to(device)
            phoneme_vector = phoneme_vector.to(device)
            decoder_inputs = phoneme_vector.to(device)

            phoneme_pred, phoneme_pred_sequence = model(grapheme_vector, g_vec_len,p_vec_len, decoder_inputs, phoneme_vector, False)

            for j,k in zip(grapheme_vector,phoneme_pred_sequence):
                graphemes.append(data_decoder(j.cpu().numpy().tolist(),1))
                phonemes.append(data_decoder(k.cpu().numpy().tolist(),0))
     
    pd = list(zip(graphemes, phonemes))

    words,pronunciations = [], []

    for i in range(0, len(pd)):
        word = "".join(str(word) for word in graphemes[i])
        phones =  " ".join(str(p) for p in phonemes[i])
        words.append(word)
        pronunciations.append(phones)
    return words,pronunciations 

def DataLoading_forG2P(text_strings):
    g2p_start = {"a": "A:", "b": "b", "c":"k", "d":"d", "e":"E", "f":"f", "g":"x", "h":"h_", "i":"i",
    "j": "j", "k":"k", "l":"l", "m":"m", "n":"n", "o":"O", "p":"p", "q":"k", "r":"r", "s":"s",
    "t":"t", "u":"9y", "v":"f", "w":"v", "x":"z", "y":"@i", "z":"z", "\'":"A:" }
    def sortingWP (d):
        words, w, p = [], [], []
        for i in range(0, len(d)):
            words.append(d[i])
            w.append(" ".join(d[i])) 
            p.append(g2p_start[d[i][0]]) # #Starting phone based on starting graph
        # print(f"w:{words}")
        # print(f"g:{w}")
        # print(f"p:{p}")
        return words, w, p
    text_strings = text_strings.lower()
    text_strings = text_strings.replace("-", " ")
    text_strings = text_strings.replace("+", " ")
    text_strings = text_strings.replace("%", "persent")
    text_strings = text_strings.replace('\"', "")
    text_strings = text_strings.replace('\'', "")
    text_strings = text_strings.replace("!", "")
    text_strings = text_strings.replace("?", "")
    text_strings = text_strings.replace(".", "")
    text_strings = text_strings.replace(",", "")
    train_data_lines = split_nums_letters(text_strings)
    words, graphemes, phonemes = sortingWP(train_data_lines)
    
    return words, graphemes, phonemes

def padding_data(batch):

    #Each sequence has a form:
    # grapheme_vector, phoneme_vector, decoder_inputs, g_vec_len, p_vec_len, graphemes, phonemes

    # def get_components(batch, index):
    #     ans = []
    #     for i in batch:
    #         ans.append(i[index])
    #     return ans
    
    def pad_seq(batch, index, max_len):
        ans = []
        no_zeros_to_add = 0
        for i in batch:
            no_zeros_to_add = max_len - len(i[index])
            ans.append(i[index] + [0] * no_zeros_to_add)
        return torch.LongTensor(ans)
    
    grapheme_lens = [len(g[0]) for g in batch]

    phonemes_lens = [len(p[1]) for p in batch]

    input_maxlen = max(grapheme_lens)
    output_maxlen = max(phonemes_lens)
    padded_inputs = pad_seq(batch, 0, input_maxlen)
    padded_outputs = pad_seq(batch, 1, output_maxlen)
    padded_decoder_inputs = pad_seq(batch, 2, output_maxlen)

    return padded_inputs, padded_outputs, padded_decoder_inputs, grapheme_lens, phonemes_lens

def cfg_init():
    class Config:
        seed = '5'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 200
        batch_size = 128
        hidden_dim = 256
        embed_dim = 256
        dropout = 0.1
        dec_max_len = 30
        MAX_LENGTH = 20
        teacher_forcing_ratio = 0.5
        n_layers = 2
        lr = 0.001
        graphemes = ['<pad>', '<unk>', '</s>', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'ä', 'è', 'é', 'ê', 'ë', 'í', 'ï', 'ó', 'ô', 'ö', 'ú', 'û']
        phonemes = ['<pad>', '<unk>', '<s>', '</s>', '2:', '9', '9y', '@', '@i', '@u', 'A:', 'E', 'N', 'O', 'Of', 'S', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h_', 'i', 'i@', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'u@', 'v', 'w', 'x', 'y', 'z', '{']   
        graph2index = {g: idx for idx, g in enumerate(graphemes)}
        index2graph = {idx: g for idx, g in enumerate(graphemes)}

        phone2index = {p: idx for idx, p in enumerate(phonemes)}
        index2phone = {idx: p for idx, p in enumerate(phonemes)}
        g_vocab_size = len(graphemes)
        p_vocab_size = len(phonemes)
    
    cfg = Config()
    return cfg

def check_if_in_dict(afr_dict, words, graphemes, phonemes):
    word_list, g_field, p_field = [], [], []
    dict_entries = afr_dict.keys()
    
    for i, word in enumerate(words):
        
        if word not in dict_entries:
            word_list.append(word)
            g_field.append(graphemes[i])
            p_field.append(phonemes[i])

    return word_list, g_field, p_field




def process_text_input(model, afr_dict, text_input):
    cfg = cfg_init()
    data_set_words, data_set_G, data_set_P = DataLoading_forG2P(text_input)
    word_set, data_set_G, data_set_P = check_if_in_dict(afr_dict, data_set_words, data_set_G, data_set_P)
    dataset_to_g2p = G2PData(data_set_G, data_set_P)
    dataset_to_g2p_iter =  data.DataLoader(dataset_to_g2p, batch_size=cfg.batch_size, shuffle=False, collate_fn=padding_data)
    graphemes, phonemes = generate_pd(model, dataset_to_g2p_iter, cfg.device)
    for g,p in zip(graphemes,phonemes):
        afr_dict[g] = p
    #return graphemes, phonemes


