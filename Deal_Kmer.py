import numpy as np
import collections
from getSequence import *
import numpy as np
from sklearn.model_selection import train_test_split

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index   


def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index    


def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]        
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  

def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**4
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        n=n//base
        ch3=chars[n%base]          
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))  
    return  word_index  


def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict


def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'U')]] = 1
    return vectors.tolist()



coden_dict1 = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
              'UGU': 1, 'UGC': 1,  # systeine<C>
              'GAU': 2, 'GAC': 2,  # aspartic acid<D>
              'GAA': 3, 'GAG': 3,  # glutamic acid<E>
              'UUU': 4, 'UUC': 4,  # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
              'CAU': 6, 'CAC': 6,  # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
              'AAA': 8, 'AAG': 8,  # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
              'AUG': 10,  # methionine<M>
              'AAU': 11, 'AAC': 11,  # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
              'CAA': 13, 'CAG': 13,  # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
              'UGG': 18,  # tryptophan<W>
              'UAU': 19, 'UAC': 19,  # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
              }
coden_dict2 = {'A': 0, 'G': 1, 'C': 2, 'T': 3,'U': 3,'AA':4,'AG':5,'AC':6,'AT':7,'GA':8,'GG':9,'GC':10,'GT':11,'CA':12,'CG':13,'CC':14,'CT':15
,'TA':16,'TG':17,'TC':18,'TT':19}

def coden1(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq) - 2):
        vectors[i][coden_dict1[seq[i:i + 3].replace('T', 'U')]] = 1
    return vectors.tolist()
def coden2(seq):
    vectors = np.zeros((len(seq), 1))
    for i in range(len(seq)):
        vectors[i]=coden_dict2[seq[i]]
    return vectors.tolist()
def coden3(seq):
    vectors = np.zeros((len(seq), 1))
    for i in range(len(seq)-1):
        vectors[i]=coden_dict2[seq[i:i + 2]]
    return vectors.tolist()

def dealwithSequence(protein):
    dataX = []
    dataY = []
    with open('./Datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                dataX.append(coden1(line.strip()))
                dataY.append([0, 1])
    with open('./Datasets/circRNA-RBP/' +  protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                dataX.append(coden1(line.strip()))
                dataY.append([1, 0])

    dataX = np.array(dataX)
    return dataX



def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T') 
    print(seq)
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)

    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
    print(new_array)
    return new_array

def dealwithdata(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    dataX = []
    with open('./Datasets/circRNA-RBP/' + protein + '/positive') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                acid = coden1(line.strip())
                Kmer = np.hstack((kmer1,kmer2,acid))
                dataX.append(Kmer.tolist())
    with open('./Datasets/circRNA-RBP/' + protein + '/negative') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(), 2, tris2)
                acid = coden1(line.strip())
                Kmer = np.hstack((kmer1, kmer2, acid))
                dataX.append(Kmer.tolist())
    dataX = np.array(dataX)
    return dataX