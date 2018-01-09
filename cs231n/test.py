def my_enumerate(li):
	cur_idx = 0
	while cur_idx < len(li):
		yield cur_idx, li[cur_idx]
		cur_idx+=1

li = [0, 1, 2, 3, 4]
enum = my_enumerate(li)
while True:
	try:
		next_idx, next_val = next(enum)
		print(next_idx, next_val)
	except StopIteration:
		break