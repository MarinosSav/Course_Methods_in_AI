for line in open('dialog_acts.dat', 'r'):
    item = line.rstrip()
    action = item.split(' ', 1)[0]
    utterance = item.replace(action, "")
    print(item, utterance)

