import numpy as np
import pandas as pd

#  crispr-ip 7*24
def crispr_ip_coding(target_seq, off_target_seq):
    encoded_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1], '_': [0, 0, 0, 0],
                    '-': [0, 0, 0, 0]}
    pos_dict = {'A': 1, 'T': 2, 'G': 3, 'C': 4, '_': 5, '-': 5}
    tlen = 24
    target_seq = "-" * (tlen - len(target_seq)) + target_seq.upper()
    off_target_seq = "-" * (tlen - len(off_target_seq)) + off_target_seq.upper()

    target_seq = list(target_seq)
    off_target_se = list(off_target_seq)

    for i in range(len(target_seq)):
        if target_seq[i] == 'N':
            target_seq[i] = off_target_seq[i]


    target_seq_code = np.array([encoded_dict[base] for base in target_seq])
    off_target_seq_code = np.array([encoded_dict[base] for base in off_target_se])
    on_off_dim6_codes = []
    for i in range(len(target_seq)):
        diff_code = np.bitwise_or(target_seq_code[i], off_target_seq_code[i])
        dir_code = np.zeros(2)
        if pos_dict[target_seq[i]] == pos_dict[off_target_seq[i]]:
            diff_code = diff_code * -1
            dir_code[0] = 1
            dir_code[1] = 1
        elif pos_dict[target_seq[i]] < pos_dict[off_target_seq[i]]:
            dir_code[0] = 1
        elif pos_dict[target_seq[i]] > pos_dict[off_target_seq[i]]:
            dir_code[1] = 1
        else:
            raise Exception("Invalid seq!", target_seq, off_target_seq)
        on_off_dim6_codes.append(np.concatenate((diff_code, dir_code)))
    on_off_dim6_codes = np.array(on_off_dim6_codes)
    isPAM = np.zeros((24, 1))
    isPAM[-3:, :] = 1
    on_off_code = np.concatenate((on_off_dim6_codes, isPAM), axis=1)
    return on_off_code

class Encoder():
    def __init__(self, on_seq, off_seq):
        tlen = 24
        self.on_seq = "-" *(tlen-len(on_seq)) +  on_seq.upper()
        self.off_seq = "-" *(tlen-len(off_seq)) + off_seq.upper()
        self.encoded_dict_indel = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0],
                                   'G': [0, 0, 1, 0, 0], 'C': [0, 0, 0, 1, 0], '_': [0, 0, 0, 0, 1], '-': [0, 0, 0, 0, 0]}
        self.direction_dict = {'A':5, 'G':4, 'C':3, 'T':2, '_':1}
        self.encode_on_off_dim7()

    def encode_sgRNA(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        sgRNA_bases = list(self.on_seq)

        for i in range(len(sgRNA_bases)):
            if sgRNA_bases[i] == "N":
                sgRNA_bases[i] = list(self.off_seq)[i]
            code_list.append(encoded_dict[sgRNA_bases[i]])
        self.sgRNA_code = np.array(code_list)

    def encode_off(self):
        code_list = []
        encoded_dict = self.encoded_dict_indel
        off_bases = list(self.off_seq)
        for i in range(len(off_bases)):
            code_list.append(encoded_dict[off_bases[i]])
        self.off_code = np.array(code_list)

    def encode_on_off_dim7(self):
        self.encode_sgRNA()
        self.encode_off()
        on_bases = list(self.on_seq)
        off_bases = list(self.off_seq)
        on_off_dim7_codes = []
        for i in range(len(on_bases)):
            diff_code = np.bitwise_or(self.sgRNA_code[i], self.off_code[i])
            on_b = on_bases[i]
            off_b = off_bases[i]
            if on_b == "N":
                on_b = off_b
            if off_b == "N":
                off_b = on_b
            dir_code = np.zeros(2)
            if on_b == "-" or off_b == "-" or self.direction_dict[on_b] == self.direction_dict[off_b]:
                pass
            else:
                if self.direction_dict[on_b] > self.direction_dict[off_b]:
                    dir_code[0] = 1
                else:
                    dir_code[1] = 1
            on_off_dim7_codes.append(np.concatenate((diff_code, dir_code)))
        self.on_off_code = np.array(on_off_dim7_codes)

#24*7
def crispr_net_coding(on_seq,off_seq):

    e = Encoder(on_seq=on_seq, off_seq=off_seq)
    return e.on_off_code

# 23*4
def cnn_predict(guide_seq, off_seq):

    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]

    if len(off_seq) == 24:
        off_seq = off_seq[1:]

    # print(guide_seq+"："+off_seq)

    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    pair_code = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]

        gRNA_list[i] = gRNA_list[i].upper()
        off_list[i] = off_list[i].upper()

        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
    input_code = np.array(pair_code).reshape(1, 1, 23, 4)
    return input_code

#23*14
def dnt_coding(on_seq, off_seq):
    on_seq = on_seq.upper()
    off_seq = off_seq.upper()

    on_seq = list(on_seq)
    off_seq = list(off_seq)

    for i in range(len(off_seq)):
        if on_seq[i] == 'N':
            on_seq[i] = off_seq[i]

        if off_seq[i] == 'N':
            off_seq[i] = on_seq[i]

    on_seq = ''.join(on_seq)
    off_seq = ''.join(off_seq)

    def one_hot_encode_seq(data):
        if len(data)==24:
            data = data[1:]
        alphabet = 'AGCT'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in data]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    arr1 = one_hot_encode_seq(on_seq)
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(off_seq)
    arr2 = np.asarray(arr2).T
    combined = np.concatenate((arr1, arr2))

    encoded_list = np.zeros((5, 23))
    for m in range(23):
        arr1 = combined[0:4, m].tolist()
        arr2 = combined[4:8, m].tolist()
        arr = []
        if arr1 == arr2:
            arr = [0, 0, 0, 0, 0]
        else:
            arr = np.add(arr1, arr2).tolist()
            arr.append(1 if (arr == [1, 1, 0, 0] or arr == [0, 0, 1, 1]) else -1)
        encoded_list[:, m] = arr

    position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    final_encoding = np.zeros((14, 23))
    for k in range(8):
        final_encoding[k] = combined[k]
    final_encoding[8:13] = encoded_list[0:5]
    final_encoding[13] = position

    final_encoding = final_encoding.reshape(14, 23).transpose((1,0))
    return final_encoding


def dnt_coding_24_14(on_seq, off_seq):
    on_seq = on_seq.upper()
    off_seq = off_seq.upper()

    on_seq = list(on_seq)
    off_seq = list(off_seq)

    for i in range(len(off_seq)):
        if on_seq[i] == 'N':
            on_seq[i] = off_seq[i]

        if off_seq[i] == 'N':
            off_seq[i] = on_seq[i]

        if on_seq[i] == '-':
            on_seq[i] = '_'

        if off_seq[i] == '-':
            off_seq[i] = '_'
    on_seq = ''.join(on_seq)
    off_seq = ''.join(off_seq)

    def one_hot_encode_seq(data):
        # if len(data)==24:
        #     data = data[1:]
        alphabet = 'AGCT_'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[char] for char in data]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        return onehot_encoded

    arr1 = one_hot_encode_seq(on_seq)
    arr1 = np.asarray(arr1).T
    arr2 = one_hot_encode_seq(off_seq)
    arr2 = np.asarray(arr2).T
    combined = np.concatenate((arr1, arr2))

    encoded_list = np.zeros((5, 24))
    for m in range(24):
        arr1 = combined[0:4, m].tolist()
        arr2 = combined[4:8, m].tolist()
        arr = []
        if arr1 == arr2:
            arr = [0, 0, 0, 0, 0]
        else:
            arr = np.add(arr1, arr2).tolist()
            arr.append(1 if (arr == [1, 1, 0, 0] or arr == [0, 0, 1, 1]) else -1)
        encoded_list[:, m] = arr

    # 加区域分开编码
    position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    final_encoding = np.zeros((14, 24))
    for k in range(8):
        final_encoding[k] = combined[k]
    final_encoding[8:13] = encoded_list[0:5]
    final_encoding[13] = position

    final_encoding = final_encoding.reshape(14, 24).transpose((1,0))
    return final_encoding



encoding_map = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11,
    'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15,
    '-A': 16, '-C': 17,'-G': 18, '-T': 19,
    'A-': 20, 'C-': 21,'G-': 22, 'T-': 23,
    '--': 24
}
def word_Encoding(sgRNA, DNA):

    if len(sgRNA) == 24:
        sgRNA = sgRNA[1:]

    if len(DNA) == 24:
        DNA = DNA[1:]

    sgRNA = list(sgRNA)

    sgRNA[-3] = DNA[-3]
    DNA = list(DNA)
    for j in range(len(sgRNA)):
        if sgRNA[j] == 'N':
            sgRNA[j] = DNA[j]
        if DNA[j] == 'N':
            DNA[j] = sgRNA[j]

    pairs = [(sgRNA[i].upper() if sgRNA[i] != '_' else '-') + (DNA[i].upper() if DNA[i] != '_' else '-') for i in range(len(sgRNA))]

    return [encoding_map[p] for p in pairs]



def dl_offtarget(guide_seq, off_seq):
    """
    Modifies the encoding function to concatenate the encoded guide_seq and off_seq
    into a 23x8 matrix.
    """

    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]

    if len(off_seq) == 24:
        off_seq = off_seq[1:]

    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    gRNA_encoded = []
    off_encoded = []
    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_list[i] = gRNA_list[i].upper()
        off_list[i] = off_list[i].upper()

        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        gRNA_encoded.append(gRNA_base_code)
        off_encoded.append(DNA_based_code)

    # Concatenate the encoded guide sequence and off-target sequence
    concatenated_code = np.concatenate((gRNA_encoded, off_encoded), axis=1)
    input_code = concatenated_code.reshape(1, 1, 23, 8)
    return input_code


def dl_crispr(guide_seq, off_seq):
    """
    Extends the encoding function to concatenate the encoded guide_seq and off_seq into a 23x8 matrix,
    and then further concatenates a 23x12 matrix to represent mismatch types between guide RNA and DNA sequences.
    """

    # Adjust sequences if they start with a length of 24
    if len(guide_seq) == 24:
        guide_seq = guide_seq[1:]
    if len(off_seq) == 24:
        off_seq = off_seq[1:]

    guide_seq = guide_seq[:-3]
    off_seq = off_seq[:-3]

    # Encoding dictionary for A, T, G, C
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}

    # Mismatch encoding dictionary
    mismatch_dict = {
        'AC': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'AG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'AT': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'CA': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'CG': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'CT': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'GA': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'GC': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'GT': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

    gRNA_list = list(guide_seq.upper())
    off_list = list(off_seq.upper())
    gRNA_encoded = []
    off_encoded = []
    mismatch_encoded = []

    for i in range(len(gRNA_list)):
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        gRNA_encoded.append(gRNA_base_code)
        off_encoded.append(DNA_based_code)

        # Encode mismatches
        mismatch_type = gRNA_list[i] + off_list[i]
        mismatch_code = mismatch_dict.get(mismatch_type, [0] * 12)  # Default to no mismatch
        mismatch_encoded.append(mismatch_code)

    # Concatenate the encoded guide sequence and off-target sequence
    concatenated_code = np.concatenate((gRNA_encoded, off_encoded), axis=1)

    # Further concatenate the mismatch encoding
    full_concatenated_code = np.concatenate((concatenated_code, mismatch_encoded), axis=1)
    input_code = full_concatenated_code.reshape(1, 1, 20, -1)  # Reshape according to your model's input requirement
    return input_code

# guide_seq = "-GGTGAGTGAGTGTGTGCGTGAGG"
# off_seq = "-G_TGTGTGTGTGTGTGAGTGAAC"
# encoded_matrix = dnt_coding_24_14(guide_seq, off_seq)
# print(encoded_matrix)