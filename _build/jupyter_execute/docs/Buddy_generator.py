#!/usr/bin/env python
# coding: utf-8

# In[24]:


import random

participants = ['Ephie',
    'Jakub',
    'Diego Nicolas',
    'Juliana',
    'Robert',
    'Anne',
    'Antonia',
    'Lara-Heléne',
    'Franziska',
    'Natalja',
    'Leonie',
    'Arne',
    'Patrice Nicola',
    'Olga',
    'Anna',
    'Ashraful',
    'Shawn Adrian',
    'Nele',
    'Naomi Shannon Heather Nahida',
    'Josefa Aletia',
    'Dominika',
    'Leonie',
    'Ronja',
    'Sara']


while len(participants)>=1:
    buddies = random.sample(participants, 2)
    print('{0} - {1} '.format(buddies[0],buddies[1]))
    for buddy in buddies:
        participants.remove(buddy)

print(participants)


# In[ ]:




