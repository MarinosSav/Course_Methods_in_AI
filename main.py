
records = []
training_set = []
test_set = []
percent = 0.85

for line in open('dialog_acts.dat', 'r'):
    item = line.rstrip()
    action = item.split(' ', 1)[0]
    utterance = item.replace(action, "")
    records.append((action, utterance))

for i in range(len(records)):
    if i < percent * len(records):
        training_set.append(records[i])
    else:
        test_set.append(records[i])

#for l in range(len(test_set)):
#    print(l, "action:", test_set[l][0], "utterance:", test_set[l][1])