"""
COPYRIGHT : https://gist.github.com/karpathy/d4dee566867f8291f086 in cs231n
"""
import numpy as np

# data input / output
datafile_dir = "input.txt"
data = open(datafile_dir, 'r', encoding="UTF8").read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("Data has {0} characters, {1} unique.".format(data_size, vocab_size))
char_to_ix = { ch : i for i,ch in enumerate(chars)}
ix_to_char = { i : ch for i,ch in enumerate(chars)}

# hyperparameters
hidden_size = 100 # size of hidden layer of neuron
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-2

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size,1)) # output bias

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # forward
    for t in range(len(inputs)):
        # xs는 dict와 list로 구성된 input 데이터의 원-핫 코딩 형태라고 생각하면 됨
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]]=1
        # Wxh : 100x59 matrix / xs[t] : 59x1 / Whh : 100x100 / hs[t-1] : 100x1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        # ps 딕셔너리의 "t" 키에 해당하는 값에 로그. 뒤에 붙은 0은 ps[t][targets[t]]가 길이 1짜리 리스트라 요소만 빼서 list -> 숫자로 변환하는 캐스팅 효과를 주기 위함임.
        loss += -np.log(ps[t][targets[t],0])

    # backward
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    # hidden state 끼리의 역전파량
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        # dy : 59x1 / hs[t] : 100x1 / Why : 59x100
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t]*hs[t])*dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]

def sample(h, seed_ix, n):
    """
    지금까지 학습된 파라미터를 바탕으로 길이 n만큼의 정수(여기서는 현재 문자의 key = integer)를 추출함
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        # Wxh : 100x59 matrix / x : 59X1 vector / Whh : 100x100 matrix / h : 100x1
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        # Why : 59x100 / y = 59x1 (59 : vocab size)
        y = np.dot(Why, h) + by
        p = np.exp(y)/np.sum(np.exp(y))
        # p의 확률분포로 0~58까지 중 1개를 추출함
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size,1))
        x[ix] = 1
        ixes.append(ix)

    return ixes

# Training & Prediction
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while True :

    # prepare inputs
    if p+seq_length+1 >= len(data) or n == 0:
        # hidden size : 100
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    # target : 각 인풋 요소의 다음에 위치한 요소(요소의 인덱스)
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    if n%100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        #print("----\n {0} \n----".format(txt))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if n%100 == 0 :
        print("iter : {0}, loss : {1}".format(n, loss))
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter

