#!/usr/bin/env python
# coding: utf-8

# (python:solution)=
# # Python exercises
# 
# Here you find a set of exercises to revisit your python abilities.

# Exercise 01: Create a function that takes an integer as an argument and returns "Even" for even numbers or "Odd" for odd numbers.

# In[2]:


def even_or_odd(number):
    return 'Odd' if number % 2 else 'Even'


# Test the function with the following examples:

# In[4]:


try:
    assert even_or_odd(2) == "Even"
    print("Test 1: Correct")
except AssertionError:
    print("Test 1: Incorrect")

try:
    assert even_or_odd(1) == "Odd"
    print("Test 2: Correct")
except AssertionError:
    print("Test 2: Incorrect")

try:
    assert even_or_odd(12) == "Even"
    print("Test 3: Correct")
except AssertionError:
    print("Test 3: Incorrect")


# Exercise 02: This code does not execute properly. Try to figure out why.

# In[5]:


def multiply(a, b):
    return a * b


# In[6]:


try:
    assert multiply(2,3) == 6
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 03: You get an array of numbers, return the sum of all of the positives ones.
# 
# Example [1,-4,7,12] => 1 + 7 + 12 = 20
# 
# Note: if there is nothing to sum, the sum is default to 0.

# In[7]:


def positive_sum(arr):
    return sum(x for x in arr if x > 0)


# In[8]:


try:
    assert positive_sum([1,2,3,4,5]) == 15
    print("Test 1: Correct")
except AssertionError:
    print("Test 1: Incorrect")
try:
    assert positive_sum([-1,2,3,4,-5]) == 9
    print("Test 2: Correct")
except AssertionError:
    print("Test 2: Incorrect")


# Exercise 04: Given an array of integers your solution should find the smallest integer.
# 
# For example:
# 
# - Given [34, 15, 88, 2] your solution will return 2
# - Given [34, -345, -1, 100] your solution will return -345
# 
# You can assume, for the purpose of this exercise, that the supplied array will not be empty.

# In[9]:


def find_smallest_int(arr):
    return min(arr)


# In[10]:


try:
    assert find_smallest_int([35,15,88,2]) == 2
    print("Test 1: Correct")
except AssertionError:
    print("Test 1: Incorrect")
try:
    assert find_smallest_int([34,-345,-1,100]) == -345
    print("Test 2: Correct")
except AssertionError:
    print("Test 2: Incorrect")


# Exercise 05: Your task is to create a function that does four basic mathematical operations.
# 
# The function should take three arguments - operation(string/char), value1(number), value2(number) and should return result of numbers after applying the chosen operation.
# 
# *Examples (Operator, value1, value2) --> output*
# 
# ```('+', 4, 7) --> 11
# ('-', 15, 18) --> -3
# ('*', 5, 5) --> 25
# ('/', 49, 7) --> 7
# ```

# In[13]:


def basic_op(operator, value1, value2):
    if operator=='+':
       return value1+value2
    if operator=='-':
       return value1-value2
    if operator=='/':
       return value1/value2
    if operator=='*':
       return value1*value2


# In[14]:


try:
    assert basic_op('-',15,18) == -3
    print("Test 1: Correct")
except AssertionError:
    print("Test 1: Incorrect")
try:
    assert basic_op('*',5,5) == 25
    print("Test 2: Correct")
except AssertionError:
    print("Test 2: Incorrect")


# Exercise 06: In this kata you will create a function that takes in a list and returns a list with the reverse order.
# 
# Examples (Input -> Output)
# 
# ```
# * [1, 2, 3, 4]  -> [4, 3, 2, 1]
# * [9, 2, 0, 7]  -> [7, 0, 2, 9]
# ```

# In[15]:


def reverse_list(l):
    return l[::-1]


# In[16]:


try:
    assert reverse_list([1, 2, 3, 4]) == [4, 3, 2, 1]
    print("Test 1: Correct")
except AssertionError:
    print("Test 1: Incorrect")
try:
    assert reverse_list([9, 2, 0, 7]) == [7, 0, 2, 9]
    print("Test 2: Correct")
except AssertionError:
    print("Test 2: Incorrect")


# Exercise 07: Write a function that computes the volume of a sphere given its radius.

# In[17]:


def vol(rad):
    return (4/3)*(3.14)*(rad**3)


# In[18]:


try:
    assert vol(2) == 33.49333333333333
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 08: Write a function that checks whether a number is in a given range (inclusive of high and low)

# In[19]:


def ran_bool(num,low,high):
    #Check if num is between low and high (including low and high)
    return num in range(low,high+1)


# In[20]:


try:
    assert ran_bool(5,2,7) == True
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 09: Write a Python function that takes a list and returns a new list with unique elements of the first list.
# 
# ```
# Sample List : [1,1,1,1,2,2,3,3,3,3,4,5]
# Unique List : [1, 2, 3, 4, 5]
# ```

# In[21]:


def unique_list(lst):
    # Also possible to use list(set())
    x = []
    for a in lst:
        if a not in x:
            x.append(a)
    return x


# In[22]:


try:
    assert unique_list([1,1,1,1,2,2,3,3,3,3,4,5]) == [1, 2, 3, 4, 5]
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 10: Write a Python function to multiply all the numbers in a list.
# 
# ```
# Sample List : [1, 2, 3, -4]
# Expected Output : -24
# ```

# In[23]:


def multiply(numbers):
    total = 1
    for x in numbers:
        total *= x
    return total


# In[24]:


try:
    assert multiply([1,2,3,-4]) == -24
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 11: Write a Python funtion that takes a list and takes the differences between the elemtens
# 
# ```
# Sample List: [1,3,7,5,5,2,0,1,2]
# Expected Output: [2,4,-2,0,-3,-2,1,1]
# ```
# 
# You can either use simple list functions or numpy arrays ... It's up to you!

# In[25]:


import numpy as np

def diff_list(lst):
    lst = np.array(lst)
    return list(np.subtract(lst[1:],lst[:-1]))


# In[26]:


try:
    assert diff_list([1,3,7,5,5,2,0,1,2]) == [2,4,-2,0,-3,-2,1,1]
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# Exercise 12: Some numpy exercises

# **Write a NumPy program to create an array with values ranging from a to b.**
# 
# ```
# Example: a=12, b=17 --> [12,13,14,15,16,17]
# ```

# In[27]:


import numpy as np

def rng(a, b):
    return np.arange(a, b)


# In[28]:


try:
    assert (rng(12,17) == np.array([12,13,14,15,16])).all()
    print("Test: Correct")
except AssertionError:
    print("Test: Incorrect")


# **Write two NumPy programs to convert the values of Centigrade degrees into Fahrenheit degrees and vice versa. Values are stored into a NumPy array.**
# 
# Remember: 
# ```
# Celsius to Fahrenheit (9C + (32*5))/5
# Fahrenheit to Celsius 5*(F-32))/9
# ```
# 
# Sample Array:
# ```
# Values in Fahrenheit degrees [0, 12, 45.21, 34, 99.91]
# Values in Centigrade degrees [-17.78, -11.11, 7.34, 1.11, 37.73, 0. ]
# ```

# In[29]:


import numpy as np

def fahr2celsius(lst):
    F = np.array(lst)
    print("Values in Fahrenheit degrees:")
    print(F)
    print("Values in  Centigrade degrees:")
    print(np.round((5*F/9 - 5*32/9),2))
    return (np.round((5*F/9 - 5*32/9),2))


def celsius2fahr(lst):
    C = np.array(lst)
    print("Values in Celsius degrees:")
    print(C)
    print("Values in  Centigrade degrees:")
    print(np.round((9*C+(32*5))/5,2))
    return (np.round((9*C+(32*5))/5,2))


# In[30]:


try:
    assert (fahr2celsius([0, 12, 45.21, 34, 99.91, 32]) == np.array([-17.78,-11.11,7.34,1.11,37.73,0.])).all()
    print("\n Test: Correct")
except AssertionError:
    print("\n Test: Incorrect")

try:
    assert (celsius2fahr([-17.78,-11.11,7.34,1.11,37.73,0.]) == np.array([0, 12, 45.21, 34, 99.91, 32])).all()
    print("\n Test: Correct")
except AssertionError:
    print("\n Test: Incorrect")

