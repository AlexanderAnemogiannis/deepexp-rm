import numpy as np

def get_llq_action(machine):
    ''' return index of least-loaded queue'''
    llq = np.argmax(machine.avbl_slot)
    return llq

def get_random_action(pa):
    ''' randomly select a queue to populate '''
    act = np.random.randint(0, pa.num_q)
    return act
