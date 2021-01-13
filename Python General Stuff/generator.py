def square(nums):
    for i in nums:
        yield i * i

my_nums = square(range(6))

my_nums
next(my_nums)
next(my_nums)
next(my_nums)
next(my_nums)

for num in my_nums:
    print(num)


# Comprehensions to create generator

my_nums = (x * x for x in range(6))
my_nums
list(my_nums)