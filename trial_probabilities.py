# 16: 0, 17: 0.140185, 18: 0.134378, 19: 0.129917, 20: 0.12391, 21: 0.117626, 22: 0.151463, 23: 0.062204, 24: 0.055041, 25: 0.046757, 26: 0.038519

# For dealer start card = 2
def dealer_2():
    dealer = [0.0] * 27
    dealer[17] = 0.140185
    dealer[18] = 0.134378
    dealer[19] = 0.129917
    dealer[20] = 0.12391
    dealer[21] = 0.117626
    dealer[22] = 0.151463
    dealer[23] = 0.062204
    dealer[24] = 0.055041
    dealer[25] = 0.046757
    dealer[26] = 0.038519
    return dealer

# For dealer start card = ACE
def dealer_ace():
    dealer = [0.0] * 27
    dealer[17] = 0.130662
    dealer[18] = 0.130764
    dealer[19] = 0.130579
    dealer[20] = 0.130585
    dealer[21] = 0.362077
    dealer[22] = 0.031203
    dealer[23] = 0.027179
    dealer[24] = 0.023132
    dealer[25] = 0.018995
    dealer[26] = 0.014824
    return dealer


dealer = dealer_ace()
player = [0.0] * 27
player[16] = 1.0/13 # ace
player[17] = 1.0/13 # 2
player[18] = 1.0/13 # 3
player[19] = 1.0/13 # 4
player[20] = 1.0/13 # 5
player[21] = 1.0/13 # 6
player[22] = 1.0/13 # 7
player[23] = 1.0/13 # 8
player[24] = 1.0/13 # 9
player[25] = 4.0/13 # 10

player = [0.0] * 27
player[17] = 1.0

# assert sum(player) == 1.0 and sum(dealer) == 1

win_prob = 0
draw_prob = 0
loss_prob = 0
for p in range(12, 26):
    for d in range(17, 27):
        prob = player[p] * dealer[d]
        if p > 21:
            loss_prob += prob
        elif d > 21:
            win_prob += prob
        elif p > d:
            win_prob += prob
        elif p == d:
            draw_prob += prob
        else:
            loss_prob += prob

total = win_prob + draw_prob + loss_prob
print("w: {}, d: {}, l: {} (t: {})".format(win_prob, draw_prob, loss_prob, total))

